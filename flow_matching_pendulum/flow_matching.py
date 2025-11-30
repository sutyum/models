"""
Core Flow Matching Implementation

This module implements the Flow Matching algorithm for generative modeling.
Flow matching learns a continuous-time flow that transforms a simple source
distribution (Gaussian noise) into a complex target distribution (expert actions).

Key Papers:
- Flow Matching for Generative Modeling (Lipman et al., 2023)
- Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math


class FlowMatching:
    """
    Flow Matching for Continuous Normalizing Flows (CNFs).

    The core idea:
    1. We want to learn a flow from noise z_0 ~ N(0,I) to data x_1 ~ p_data
    2. We parameterize this as an ODE: dz/dt = v_θ(z, t)
    3. We train v_θ to match the velocity of an optimal transport path

    For conditional flow matching (our case):
    - Source: Gaussian noise z_0
    - Target: Expert action x_1 conditioned on state s
    - Path: Linear interpolation z_t = (1-t)z_0 + t·x_1
    - Optimal velocity: v* = x_1 - z_0 (constant along the path!)
    """

    @staticmethod
    def sample_flow_path(
        target_data: torch.Tensor,
        t: float,
        device: torch.device = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a point along the optimal transport flow path.

        The path is a linear interpolation from noise to data:
            z_t = (1-t)·z_0 + t·x_1

        Where:
            z_0 ~ N(0, I) - source distribution (Gaussian noise)
            x_1 = target_data - target distribution (expert action)
            t ∈ [0, 1] - time parameter

        Args:
            target_data: Target sample (e.g., expert action), shape [..., action_dim]
            t: Time parameter in [0, 1]
            device: Device to create tensors on

        Returns:
            z_t: Interpolated sample at time t
            velocity: Optimal velocity v* = x_1 - z_0 (what the model should predict)
        """
        if device is None:
            device = target_data.device

        # Sample Gaussian noise with same shape as target
        z_0 = torch.randn_like(target_data, device=device)
        x_1 = target_data

        # Linear interpolation: z_t = (1-t)·z_0 + t·x_1
        # At t=0: z_t = z_0 (pure noise)
        # At t=1: z_t = x_1 (pure data)
        z_t = (1 - t) * z_0 + t * x_1

        # The optimal velocity for this path is constant!
        # It's the direction from noise to data: v* = x_1 - z_0
        velocity = x_1 - z_0

        return z_t, velocity

    @staticmethod
    def compute_loss(
        model: nn.Module,
        states: torch.Tensor,  # Shape: [batch, state_dim]
        actions: torch.Tensor,  # Shape: [batch, action_dim]
    ) -> torch.Tensor:
        """
        Compute the Flow Matching loss.

        The loss is simply:
            L(θ) = E_t,z_0,x_1 [||v_θ(z_t, t, s) - v*||²]

        Where:
            v_θ is our predicted velocity (model output)
            v* = x_1 - z_0 is the optimal velocity

        This is the Conditional Flow Matching (CFM) objective,
        which is simpler than the original Flow Matching loss!

        Args:
            model: The velocity predictor network v_θ(z_t, t, state)
            states: Batch of states (conditions)
            actions: Batch of expert actions (targets)

        Returns:
            Scalar loss value
        """
        batch_size = states.shape[0]
        device = states.device

        # Sample random time steps for each batch element
        # Uniform t ~ U[0, 1]
        t = torch.rand(batch_size, device=device)

        # For each (state, action) pair, sample a point on the flow path
        z_t_list = []
        velocity_target_list = []

        for i in range(batch_size):
            z_t, velocity_target = FlowMatching.sample_flow_path(
                actions[i], t[i].item(), device
            )
            z_t_list.append(z_t)
            velocity_target_list.append(velocity_target)

        z_t_batch = torch.stack(z_t_list)
        velocity_target_batch = torch.stack(velocity_target_list)

        # Predict velocity using our model
        # Model takes: (noisy_action, time, state) → predicted_velocity
        velocity_pred_batch = model(z_t_batch, t, states)

        # Mean squared error between predicted and optimal velocity
        loss = F.mse_loss(velocity_pred_batch, velocity_target_batch)

        return loss

    @staticmethod
    def sample_action(
        model: nn.Module,
        state: torch.Tensor,
        num_steps: int = 20,
        method: str = "euler",
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Generate an action by solving the flow ODE.

        Starting from noise z_0 ~ N(0,I), we integrate the ODE:
            dz/dt = v_θ(z, t, state)
        from t=0 to t=1 to get the final action.

        Args:
            model: The velocity predictor network
            state: Current state to condition on, shape [state_dim]
            num_steps: Number of integration steps (more = better quality, slower)
            method: Integration method ("euler" or "heun")
            device: Device to create tensors on

        Returns:
            Generated action, shape [action_dim]
        """
        if device is None:
            device = state.device

        # Infer action dimension from model
        action_dim = model.action_dim

        # Start with pure Gaussian noise
        z = torch.randn(action_dim, device=device)

        dt = 1.0 / num_steps

        for step in range(num_steps):
            t = torch.tensor([step * dt], device=device)

            if method == "euler":
                # Simple Euler integration: z_{t+dt} = z_t + dt·v_θ(z_t, t)
                velocity = model(z.unsqueeze(0), t, state.unsqueeze(0)).squeeze(0)
                z = z + dt * velocity

            elif method == "heun":
                # Heun's method (improved Euler, 2nd order)
                # k1 = v_θ(z_t, t)
                # k2 = v_θ(z_t + dt·k1, t+dt)
                # z_{t+dt} = z_t + dt·(k1 + k2)/2
                v1 = model(z.unsqueeze(0), t, state.unsqueeze(0)).squeeze(0)
                z_temp = z + dt * v1
                t_next = torch.tensor([t.item() + dt], device=device)
                v2 = model(z_temp.unsqueeze(0), t_next, state.unsqueeze(0)).squeeze(0)
                z = z + dt * (v1 + v2) / 2

            else:
                raise ValueError(f"Unknown integration method: {method}")

        return z


class VelocityNet(nn.Module):
    """
    Neural network that predicts the velocity field v_θ(z, t, state).

    This is the core learnable component. It takes:
    - z: Current noisy action (or intermediate sample)
    - t: Time parameter ∈ [0, 1]
    - state: Environment state (condition)

    And outputs:
    - velocity: Direction to move in action space

    Architecture:
    - Time embedding: t → [sin(2πt), cos(2πt), sin(4πt), cos(4πt), ...]
    - Concatenate [z, time_embedding, state]
    - Feed through MLP
    - Output velocity vector (same shape as z)
    """

    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        time_embed_dim: int = 16,
    ):
        """
        Initialize the velocity network.

        Args:
            action_dim: Dimension of action space
            state_dim: Dimension of state space
            hidden_dim: Hidden layer dimension
            num_layers: Number of MLP layers
            time_embed_dim: Dimension of time embedding (should be even)
        """
        super().__init__()

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.time_embed_dim = time_embed_dim

        # Input dimension: action + time_embedding + state
        input_dim = action_dim + time_embed_dim + state_dim

        # Build MLP layers
        layers = []

        for i in range(num_layers):
            if i == 0:
                in_dim = input_dim
            else:
                in_dim = hidden_dim

            if i == num_layers - 1:
                out_dim = action_dim  # Output velocity (same dim as action)
            else:
                out_dim = hidden_dim

            layers.append(nn.Linear(in_dim, out_dim))

        self.layers = nn.ModuleList(layers)

    def time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal time embedding.

        Similar to positional encoding in transformers.
        Maps scalar t → vector with periodic features.

        Args:
            t: Time tensor ∈ [0, 1], shape [batch] or []

        Returns:
            Time embedding vector of shape [batch, time_embed_dim] or [time_embed_dim]
        """
        # Ensure t has a batch dimension
        if t.dim() == 0:
            t = t.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        half_dim = self.time_embed_dim // 2
        # Frequencies: 2π, 4π, 8π, ..., 2^(half_dim) * π
        freqs = 2 * math.pi * torch.pow(2.0, torch.arange(half_dim, device=t.device))
        args = t.unsqueeze(-1) * freqs  # [batch, half_dim]
        # Interleave sin and cos
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if squeeze_output:
            embedding = embedding.squeeze(0)

        return embedding

    def forward(
        self,
        z: torch.Tensor,      # Noisy action, shape [batch, action_dim] or [action_dim]
        t: torch.Tensor,      # Time, shape [batch] or scalar
        state: torch.Tensor,  # State, shape [batch, state_dim] or [state_dim]
    ) -> torch.Tensor:
        """
        Predict velocity at (z, t, state).

        Args:
            z: Current sample in action space
            t: Time parameter
            state: Conditioning state

        Returns:
            Predicted velocity, shape [batch, action_dim] or [action_dim]
        """
        # Handle single sample vs batch
        if z.dim() == 1:
            z = z.unsqueeze(0)
            state = state.unsqueeze(0) if state.dim() == 1 else state
            t = t.unsqueeze(0) if t.dim() == 0 else t
            squeeze_output = True
        else:
            squeeze_output = False

        # Compute time embedding
        t_embed = self.time_embedding(t)

        # Concatenate all inputs
        x = torch.cat([z, t_embed, state], dim=-1)

        # Forward through MLP with activation
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # No activation on last layer
                x = F.silu(x)  # SiLU activation (smooth, works well for flows)

        if squeeze_output:
            x = x.squeeze(0)

        return x


# Example usage and testing
if __name__ == "__main__":
    # Test the flow matching implementation
    torch.manual_seed(0)

    # Dimensions
    action_dim = 1  # Pendulum has 1D action (torque)
    state_dim = 3   # Pendulum state: [cos(θ), sin(θ), θ_dot]
    batch_size = 32

    # Create model
    model = VelocityNet(
        action_dim=action_dim,
        state_dim=state_dim,
        hidden_dim=128,
        num_layers=3,
    )

    # Create dummy data
    device = torch.device("cpu")
    states = torch.randn(batch_size, state_dim, device=device)
    actions = torch.randn(batch_size, action_dim, device=device)

    # Compute loss
    loss = FlowMatching.compute_loss(model, states, actions)
    print(f"Initial loss: {loss.item():.4f}")

    # Sample an action
    test_state = torch.randn(state_dim, device=device)
    sampled_action = FlowMatching.sample_action(
        model, test_state, num_steps=10, device=device
    )
    print(f"Sampled action shape: {sampled_action.shape}")
    print(f"Sampled action: {sampled_action.detach().cpu().numpy()}")

    print("\n✅ Flow matching implementation working correctly!")
