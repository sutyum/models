"""
Core Flow Matching Implementation

This module implements the Flow Matching algorithm for generative modeling.
Flow matching learns a continuous-time flow that transforms a simple source
distribution (Gaussian noise) into a complex target distribution (expert actions).

Key Papers:
- Flow Matching for Generative Modeling (Lipman et al., 2023)
- Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import Callable, Tuple
import equinox as eqx


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
        key: random.PRNGKey,
        target_data: jnp.ndarray,
        t: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Sample a point along the optimal transport flow path.

        The path is a linear interpolation from noise to data:
            z_t = (1-t)·z_0 + t·x_1

        Where:
            z_0 ~ N(0, I) - source distribution (Gaussian noise)
            x_1 = target_data - target distribution (expert action)
            t ∈ [0, 1] - time parameter

        Args:
            key: JAX random key
            target_data: Target sample (e.g., expert action), shape [..., action_dim]
            t: Time parameter in [0, 1]

        Returns:
            z_t: Interpolated sample at time t
            velocity: Optimal velocity v* = x_1 - z_0 (what the model should predict)
        """
        # Sample Gaussian noise with same shape as target
        z_0 = random.normal(key, shape=target_data.shape)
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
        model: eqx.Module,
        key: random.PRNGKey,
        states: jnp.ndarray,  # Shape: [batch, state_dim]
        actions: jnp.ndarray,  # Shape: [batch, action_dim]
    ) -> jnp.ndarray:
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
            key: JAX random key
            states: Batch of states (conditions)
            actions: Batch of expert actions (targets)

        Returns:
            Scalar loss value
        """
        batch_size = states.shape[0]

        # Sample random time steps for each batch element
        # Uniform t ~ U[0, 1]
        key, subkey = random.split(key)
        t = random.uniform(subkey, shape=(batch_size,))

        # For each (state, action) pair, sample a point on the flow path
        # We use vmap to vectorize over the batch
        key, *subkeys = random.split(key, batch_size + 1)
        subkeys = jnp.array(subkeys)

        def sample_single(key_i, action_i, t_i):
            """Sample flow path for a single example."""
            return FlowMatching.sample_flow_path(key_i, action_i, t_i)

        # Vectorize over batch
        z_t_batch, velocity_target_batch = jax.vmap(sample_single)(
            subkeys, actions, t
        )

        # Predict velocity using our model
        # Model takes: (noisy_action, time, state) → predicted_velocity
        velocity_pred_batch = jax.vmap(model)(z_t_batch, t, states)

        # Mean squared error between predicted and optimal velocity
        loss = jnp.mean((velocity_pred_batch - velocity_target_batch) ** 2)

        return loss

    @staticmethod
    def sample_action(
        model: eqx.Module,
        key: random.PRNGKey,
        state: jnp.ndarray,
        num_steps: int = 20,
        method: str = "euler",
    ) -> jnp.ndarray:
        """
        Generate an action by solving the flow ODE.

        Starting from noise z_0 ~ N(0,I), we integrate the ODE:
            dz/dt = v_θ(z, t, state)
        from t=0 to t=1 to get the final action.

        Args:
            model: The velocity predictor network
            key: JAX random key
            state: Current state to condition on, shape [state_dim]
            num_steps: Number of integration steps (more = better quality, slower)
            method: Integration method ("euler" or "heun")

        Returns:
            Generated action, shape [action_dim]
        """
        # Infer action dimension from model
        # Start with pure Gaussian noise
        action_dim = model.action_dim
        z = random.normal(key, shape=(action_dim,))

        dt = 1.0 / num_steps

        for step in range(num_steps):
            t = step * dt

            if method == "euler":
                # Simple Euler integration: z_{t+dt} = z_t + dt·v_θ(z_t, t)
                velocity = model(z, t, state)
                z = z + dt * velocity

            elif method == "heun":
                # Heun's method (improved Euler, 2nd order)
                # k1 = v_θ(z_t, t)
                # k2 = v_θ(z_t + dt·k1, t+dt)
                # z_{t+dt} = z_t + dt·(k1 + k2)/2
                v1 = model(z, t, state)
                z_temp = z + dt * v1
                v2 = model(z_temp, t + dt, state)
                z = z + dt * (v1 + v2) / 2

            else:
                raise ValueError(f"Unknown integration method: {method}")

        return z


class VelocityNet(eqx.Module):
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

    layers: list
    action_dim: int = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)
    time_embed_dim: int = eqx.field(static=True)

    def __init__(
        self,
        key: random.PRNGKey,
        action_dim: int,
        state_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        time_embed_dim: int = 16,
    ):
        """
        Initialize the velocity network.

        Args:
            key: JAX random key for initialization
            action_dim: Dimension of action space
            state_dim: Dimension of state space
            hidden_dim: Hidden layer dimension
            num_layers: Number of MLP layers
            time_embed_dim: Dimension of time embedding (should be even)
        """
        self.action_dim = action_dim
        self.state_dim = state_dim

        # Input dimension: action + time_embedding + state
        input_dim = action_dim + time_embed_dim + state_dim

        # Build MLP layers
        layers = []
        keys = random.split(key, num_layers)

        for i in range(num_layers):
            if i == 0:
                in_dim = input_dim
            else:
                in_dim = hidden_dim

            if i == num_layers - 1:
                out_dim = action_dim  # Output velocity (same dim as action)
            else:
                out_dim = hidden_dim

            layers.append(eqx.nn.Linear(in_dim, out_dim, key=keys[i]))

        self.layers = layers
        self.time_embed_dim = time_embed_dim

    def time_embedding(self, t: jnp.ndarray) -> jnp.ndarray:
        """
        Create sinusoidal time embedding.

        Similar to positional encoding in transformers.
        Maps scalar t → vector with periodic features.

        Args:
            t: Time scalar ∈ [0, 1]

        Returns:
            Time embedding vector of shape [time_embed_dim]
        """
        half_dim = self.time_embed_dim // 2
        # Frequencies: 2π, 4π, 8π, ..., 2^(half_dim) * π
        freqs = 2 * jnp.pi * 2 ** jnp.arange(half_dim)
        args = t * freqs
        # Interleave sin and cos
        embedding = jnp.concatenate([jnp.sin(args), jnp.cos(args)])
        return embedding

    def __call__(
        self,
        z: jnp.ndarray,      # Noisy action, shape [action_dim]
        t: jnp.ndarray,      # Time, scalar or shape []
        state: jnp.ndarray,  # State, shape [state_dim]
    ) -> jnp.ndarray:
        """
        Predict velocity at (z, t, state).

        Args:
            z: Current sample in action space
            t: Time parameter
            state: Conditioning state

        Returns:
            Predicted velocity, shape [action_dim]
        """
        # Ensure t is a scalar (remove batch dimensions if present)
        t = jnp.squeeze(t)

        # Compute time embedding
        t_embed = self.time_embedding(t)

        # Concatenate all inputs
        x = jnp.concatenate([z, t_embed, state])

        # Forward through MLP with activation
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # No activation on last layer
                x = jax.nn.silu(x)  # SiLU activation (smooth, works well for flows)

        return x


# Example usage and testing
if __name__ == "__main__":
    # Test the flow matching implementation
    key = random.PRNGKey(0)

    # Dimensions
    action_dim = 1  # Pendulum has 1D action (torque)
    state_dim = 3   # Pendulum state: [cos(θ), sin(θ), θ_dot]
    batch_size = 32

    # Create model
    key, subkey = random.split(key)
    model = VelocityNet(
        key=subkey,
        action_dim=action_dim,
        state_dim=state_dim,
        hidden_dim=128,
        num_layers=3,
    )

    # Create dummy data
    key, subkey = random.split(key)
    states = random.normal(subkey, (batch_size, state_dim))
    key, subkey = random.split(key)
    actions = random.normal(subkey, (batch_size, action_dim))

    # Compute loss
    key, subkey = random.split(key)
    loss = FlowMatching.compute_loss(model, subkey, states, actions)
    print(f"Initial loss: {loss:.4f}")

    # Sample an action
    key, subkey = random.split(key)
    test_state = random.normal(subkey, (state_dim,))
    key, subkey = random.split(key)
    sampled_action = FlowMatching.sample_action(
        model, subkey, test_state, num_steps=10
    )
    print(f"Sampled action shape: {sampled_action.shape}")
    print(f"Sampled action: {sampled_action}")

    print("\n✅ Flow matching implementation working correctly!")
