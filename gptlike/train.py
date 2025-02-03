import os
import time
import math
import torch
from torch.nn import functional as F
import tiktoken
import numpy as np
import matplotlib.pyplot as plt
from eval.hellaswag import render_example, iterate_examples

from models.gpt2 import GPT2, GPTConfig
from dataloader import DataLoaderLite

debug = False
debug_print = print if debug else lambda *args, **kwargs: None

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 40

def get_lr(it: int) -> float:
    # 1) linear warmup for warmup iter steps
    if it <= warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learing rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

def calculate_validation_loss(model, val_loader, device, val_loss_steps):
    model.eval()
    val_loss_accum = 0.0
    
    with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        for _ in range(val_loss_steps):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)
            
            logits, loss = model(x, y)
            val_loss_accum += (loss.detach() / val_loss_steps)  # Use .item() to get scalar value

        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
    
    return val_loss_accum.item()

def generate_text(model, enc, device, max_length=32, max_return_sequences=4):
    model.eval()
    tokens = enc.encode("Hello, I am a program which")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(max_return_sequences, 1)
    x_gen = tokens.to(device)

    with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        while x_gen.size(1) < max_length:
            logits, _ = model(x_gen)
            next_token_logits = logits[:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x_gen = torch.cat((x_gen, next_token), dim=1)

    generated_sequences = []
    for i in range(max_return_sequences):
        tokens = x_gen[i, :].tolist()
        decoded = enc.decode(tokens)
        generated_sequences.append(decoded)

    return generated_sequences

def log_memory(step, location):
    if torch.cuda.is_available():
        print(f"Step {step}, {location}: "
              f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f}GB, "
              f"Cached: {torch.cuda.memory_reserved() / 1024**3:.2f}GB, "
              f"Max allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB")

# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss
def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

if __name__ == "__main__":
    # DDP
    from torch.distributed import init_process_group, destroy_process_group
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist

    # set up DDP (distributed data parallel)
    # torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    # print(ddp, os.environ.get('RANK'), os.environ.get('LOCAL_RANK'))

    # DDP
    if ddp:
      assert torch.cuda.is_available(), "for now we need CUDA for DDP"
      init_process_group(backend='nccl')
      ddp_rank = int(os.environ['RANK'])
      ddp_local_rank = int(os.environ['LOCAL_RANK'])
      ddp_world_size = int(os.environ['WORLD_SIZE'])
      device = f"cuda:{ddp_local_rank}"
      device_type = 'cuda'
      torch.cuda.set_device(device)
      master_process = ddp_rank == 0 # this process will do logging, checkpointing
    else:
      # vanilla, non-DDP run
      ddp_rank = 0
      ddp_local_rank = 0
      ddp_world_size = 1
      master_process = True
      # attempt to autodetect device
      device = "cpu"
      if torch.cuda.is_available():
          device = "cuda"
      elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
          device = "mps"
      print(f"using device: {device}")
      device_type = device

    print(f"{device=}: {ddp_rank=}, {ddp_world_size=}, {ddp_local_rank=}")

    # Set the random seed for reproducibility
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    total_batch_size = 524_288 # 2**19, ~0.5M tokens, GPT 2's batch size
    B = 4 # micro batch size (can fit on this GPU)
    T = 1024 # sequence length
    assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B*T"
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # Instantiate the data loader
    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train", master_process=master_process)
    val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train", master_process=master_process)

    # SPEED!: Use TF32
    torch.set_float32_matmul_precision("high") # Use TF32 on GPUs which support it

    # SPEED!: all numbers powers of 2
    model = GPT2(GPTConfig(vocab_size=50304, context_size=1024))
    # model = GPT2(GPTConfig()) # Baseline
    model.to(device)
    # SPEED!: Torch Compile
    use_compile = False
    if use_compile:
      model = torch.compile(model)

    if ddp: # DDP
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    # Training
    losses = []
    val_losses = []

    if master_process:
        print(f"Training for {max_steps} steps")

    # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8) # IMPROVEMENTS: Match hyperperameters to GPT3 paper details
    # IMPROVEMENTS: refactor optimizer into the model
    optimizer = raw_model.configure_optimizer(weight_decay=0.1, learning_rate=6e-4, device=device)

    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "log.txt")
    with open(log_file, "w") as f: # Opening just to empty the file
        pass

    val_interval = 20
    enc = tiktoken.get_encoding("gpt2")

    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        # Validation
        # Once in a while, evaluate our validation loss
        if step % val_interval == 0 or last_step:
            # log_memory(step, "Before validation")
            model.eval()
            val_loss = calculate_validation_loss(model, val_loader, device, val_loss_steps=val_interval)
            if master_process:
                print(f"validation loss: {val_loss:.4f}")
                val_losses.append(val_loss)
                with open(log_file, "a") as f:
                    f.write(f"Step {step} val loss: {val_loss:.4f}\n")
            log_memory(step, "After Validation")

        # Text generation
        # Once in a while generate some text completion with our model
        if step % (val_interval * 2) == 0 or last_step and (not use_compile):
          generated_texts = generate_text(model, enc, device)
          if master_process:
              for i, text in enumerate(generated_texts):
                print(f"step {step} rank {ddp_rank} sample {i}: {text}")
        
        # Hellaswag Eval
        # once in a while evaluate hellaswag
        if (step % val_interval == 0 or last_step) and (not use_compile):
            num_correct_norm = 0
            num_total = 0
            for i, example in enumerate(iterate_examples("val")):
                # only process examples where i % ddp_world_size == ddp_rank
                if i % ddp_world_size != ddp_rank:
                    continue
                # render the example into tokens and labels
                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(device)
                mask = mask.to(device)
                # get the logits
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(tokens)
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)
            # reduce the stats across all processes
            if ddp:
                num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
                dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                num_total = num_total.item()
                num_correct_norm = num_correct_norm.item()
            acc_norm = num_correct_norm / num_total
            if master_process:
                print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} hella {acc_norm:.4f}\n")

        # Training Loop
        model.train()
        optimizer.zero_grad()  # needed as pytorch accumulates gradients
        loss_accum = 0.0 # accumulate for microsteps for printing
        for microstep in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            # SPEED!: autocast to bfloat16
            if device_type == 'cuda':
              with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                  logits, loss = model(x, y)
              # import code; code.interact(locals=locals())
            else:
              logits, loss = model(x, y) # Baseline

            loss = loss / grad_accum_steps # For gradient accumulation
            loss_accum += loss.detach()
            if ddp:
                model.require_backward_grad_sync = (microstep == grad_accum_steps - 1) # Sync gradients on the last grad_accum_step
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # IMPROVEMENTS!: Normalize Gradients using norm of gradients

        # IMPROVEMENTS!: Cosine LR Scheduling
        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        # Set the learning rate based on our schedule
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.step()

        # if device_type == 'cuda':
        if ddp:
          torch.cuda.synchronize() # Wait for GPUs to complete the above queued up tasks

        t1 = time.time()
        dt = (t1 - t0) * 1000 # milliseconds
        tps = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size) / (t1 - t0)
        if master_process:
            print(f"step {step} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt = {dt:.2f} | tokens/sec: {tps:.0f}")
            with open(log_file, "a") as f:
                f.write(f"Step {step} train loss: {loss_accum.item():.4f}\n")

        losses.append(loss_accum.item())

    # Plot the loss
    # Create an array of x-values for the training loss
    steps = np.arange(len(losses))
    
    # Create an array of x-values for the validation loss
    val_steps = np.arange(0, len(losses), val_interval)
    
    # Ensure val_losses array matches the length of val_x
    val_losses = val_losses[:len(val_steps - 1)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, label='Training Loss')
    plt.plot(val_steps, val_losses, label='Validation Loss')
    
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.savefig("loss.png")

    if ddp:
        destroy_process_group()

    import sys; sys.exit(0)
