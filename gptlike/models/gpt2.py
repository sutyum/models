import math
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

debug = False
debug_print = print if debug else lambda *args, **kwargs: None

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Linear Projection to 4 * n_embed dimensions
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        # Gaussian Error Linear Unit, a slightly smoother version of ReLU
        self.act = nn.GELU(approximate="tanh")
        # Linear Projection back to n_embd dimensions
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Number of attention heads must be a factor of n_embd
        # Because we split the n_embd dimensional input into n_head parts
        # This allows us to compute attention in parallel and then concatenate
        # the results
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Linear projection matrices for the query, key, and value vectors as a single matrix
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        # Mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.context_size, config.context_size)).view(
                1, 1, config.context_size, config.context_size
            ),
        )  # 1, 1 is batch size and n_head. Why 1 head?

    def forward(self, x):
        # Calculate query, key, value hiddem states for all heads in a batch and move head forward
        # to be the the batch
        # nh is "number of heads"
        # hs is "head size"
        # and C is "number of channels" = nh * hs
        # e.g. in GPT-2 (124M), nh=12, hs=64, C=nh*hs=768 channels in the Transformer

        B, T, C = x.size()  # batch size, context size, n_embd
        debug_print(f"{B=}, {T=}, {C=}")

        qkv = self.c_attn(x)
        debug_print(qkv.shape)

        # Split qkv into q, k and v
        # n_embd * 3 -> n_embd, n_embd, n_embd
        # q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = qkv.split(self.n_embd, dim=2)
        assert (
            q.size() == k.size() == v.size()  # == (B, T, C // 3)
        ), "q, k, v size mismatch"

        # Split q, k, v into n_head parts
        # B, T, (q, k, v) -> B, T, n_head, C // n_head
        # We transpose to make the n_head dimension the second dimension
        # B, T, n_head, c (or n_embd) // n_head -> B, n_head, T, C // n_head
        hs = C // self.n_head

        k = k.view(B, T, self.n_head, hs).transpose(1, 2)  # B, nh, T, hs
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        # Why is this shape required?
        # Because we want to compute attention for each head in parallel

        # q: B, nh, T, hs
        # k: B, nh, T, hs -> k^T: B, nh, hs, T
        # q*k^T = B, nh, T, hs * B, nh, hs, T = B, nh, T, T
        k_T = k.transpose(-2, -1)
        sqrt_d_k = 1.0 * math.sqrt(q.size(-1))
        attn = (q @ k_T) * sqrt_d_k

        # # Baseline
        # # Mask out the upper half of the dot product matrix, excluding the diagonal
        # # In other words, we only want to consider the "past" context
        # attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # attn = F.softmax(attn, dim=-1)
        # 
        # # Apply the attention to the values
        # y = attn @ v  # B, nh, T, T @ B, nh, T, hs = B, nh, T, hs
        # # Weighted average of the values with the attention scores

        # SPEED!: flash attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2)  # B, T, nh, hs
        # contiguous() is required because of the transpose because it doesn't change the memory layout
        y = y.contiguous().view(B, T, C)
        # debug_print(y.shape, C.shape)

        # Linear projection back to n_embd dimensions
        y = self.c_proj(y)

        return y


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # Weighted sum of input and attention output
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    context_size: int = 1024  # block size is just the context size
    # 64 because: [PAD], [SOS], [EOS], [MASK], [UNK] + 60 characters
    vocab_size: int = 50257 # 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    debug: bool = False


class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                # word token n_embd
                # vocab_size x n_embd
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                # word position embedding
                # context_size x n_embd
                wpe=nn.Embedding(config.context_size, config.n_embd),
                # transformer layers
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),  # final layer norm
            )
        )

        # linear layer for output before softmax
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        # print(self.transformer.wte.weight.shape, type(self.transformer.wte))
        self.transformer.wte.weight = self.lm_head.weight
        # print(self.transformer.wte.weight.shape, type(self.transformer.wte))

        # initialize weights
        self.apply(self._init_weights)
        # print(self.transformer.wte.weight.shape, type(self.transformer.wte))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        # assert torch._check_tensor_all(self.transformer.wte.weight == self.lm_head.weight, "All must be identical")

    def forward(self, idx, targets=None):    
        # x: (batch_size, context_size) or (B, T)
        B, T = idx.size()  # T <= context_size

        assert T <= self.config.context_size, "Context size mismatch"

        # word position embedding
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)

        # word token embedding
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)

        # (B, T, n_embd) there is broadcasting happening here
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(  # B * T x vocab_size
                logits.view(-1, self.config.vocab_size),  # (B * T, vocab_size)
                targets.view(-1),  # (B * T)
            )

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type: str):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]

        from transformers import GPT2LMHeadModel

        print(f"Loading weights from pretrained gpt: {model_type}")

        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            # 350M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            # 774M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
            # 1558M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]

        config_args["vocab_size"] = 50257
        config_args["context_size"] = 1024

        # Create a new GPT2 model
        config = GPTConfig(**config_args)
        model = GPT2(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [key for key in sd_keys if not key.endswith(".attn.bias")]
        # debug_print(f"{sd_keys=}")

        # Init a huggingface/transformwers model
        hf_model = GPT2LMHeadModel.from_pretrained(model_type)
        # debug_print(f"{hf_model}")
        sd_hf = hf_model.state_dict()

        # Copy while ensuring all of the parameter are aligned and match in name and shape
        sd_keys_hf = sd_hf.keys()

        # Remove the attn bias as they are not parameters but used as masks (they are buffers)
        sd_keys_hf = [
            key for key in sd_keys_hf if not key.endswith(".attn.masked_bias")
        ]
        sd_keys_hf = [key for key in sd_keys_hf if not key.endswith(".attn.bias")]

        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # Openai weights use Conv1D instead of Linear, so we need to transpose the weights
        assert len(sd_keys) == len(sd_keys_hf), "Length mismatch"

        for key in sd_keys_hf:
            if any(key.endswith(w) for w in transposed):
                assert sd_hf[key].shape[::-1] == sd[key].shape, "Shape mismatch"
                with torch.no_grad():
                    sd[key].copy_(sd_hf[key].T)
            else:
                assert sd_hf[key].shape == sd[key].shape, "Shape mismatch"
                with torch.no_grad():
                    sd[key].copy_(sd_hf[key])

        return model

    def configure_optimizer(self, weight_decay, learning_rate, device):
        # Start with all the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # create optim groups, Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e, all weight tensors in matmuls + embeddings decay, all biases and layernorms dont
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed paramter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed paramter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Create AdamW optimizer and use the fused version if is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
        return optimizer
