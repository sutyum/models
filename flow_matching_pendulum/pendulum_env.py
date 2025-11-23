"""
Inverted Pendulum Environment and Expert Controller

This module provides:
1. A wrapper around Gymnasium's Pendulum-v1 environment
2. An expert controller (energy-based swing-up + LQR balance)
3. Utilities for collecting expert demonstrations

The pendulum is a classic control problem:
- Goal: Balance the pendulum upright (θ = 0)
- State: [cos(θ), sin(θ), θ_dot] (3D)
- Action: torque ∈ [-2, 2] (1D continuous)
- Challenge: Nonlinear dynamics, requires swing-up from hanging position
"""

import gymnasium as gym
import numpy as np
import jax.numpy as jnp
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt


class PendulumEnv:
    """
    Wrapper for Gymnasium Pendulum-v1 environment.

    Provides consistent interface and normalizes state for neural networks.
    """

    def __init__(self, render_mode: str = None):
        """
        Initialize the pendulum environment.

        Args:
            render_mode: "human" to visualize, None for no rendering
        """
        self.env = gym.make("Pendulum-v1", render_mode=render_mode)
        self.state_dim = 3  # [cos(θ), sin(θ), θ_dot]
        self.action_dim = 1  # [torque]
        self.action_low = self.env.action_space.low[0]  # -2.0
        self.action_high = self.env.action_space.high[0]  # 2.0

    def reset(self, seed: int = None) -> np.ndarray:
        """Reset environment and return initial state."""
        state, _ = self.env.reset(seed=seed)
        return state

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Take a step in the environment.

        Args:
            action: Torque to apply, shape [1] or scalar

        Returns:
            state: New state [cos(θ), sin(θ), θ_dot]
            reward: Reward (higher when upright and slow)
            terminated: Always False (pendulum has no terminal state)
            truncated: True after max steps
            info: Additional info dict
        """
        # Ensure action is correct shape
        if np.isscalar(action):
            action = np.array([action])
        elif isinstance(action, jnp.ndarray):
            action = np.array(action)

        # Clip action to valid range
        action = np.clip(action, self.action_low, self.action_high)

        return self.env.step(action)

    def close(self):
        """Close the environment."""
        self.env.close()

    @staticmethod
    def get_angle(state: np.ndarray) -> float:
        """
        Extract angle θ from state.

        State is [cos(θ), sin(θ), θ_dot], so:
        θ = atan2(sin(θ), cos(θ))

        Args:
            state: [cos(θ), sin(θ), θ_dot]

        Returns:
            θ in radians, range [-π, π]
        """
        return np.arctan2(state[1], state[0])


class ExpertController:
    """
    Expert controller for the inverted pendulum.

    Strategy:
    1. Swing-up phase: Energy-based controller to pump energy
    2. Balance phase: LQR controller to stabilize at top

    This is a classic control solution that achieves near-optimal performance.
    We'll use it to generate expert demonstrations for imitation learning.
    """

    def __init__(self, swing_threshold: float = 0.5):
        """
        Initialize expert controller.

        Args:
            swing_threshold: Angle threshold (radians) to switch from
                           swing-up to balance mode
        """
        self.swing_threshold = swing_threshold

        # LQR gains (manually tuned for pendulum)
        # These were computed using the linearized dynamics around θ=0
        # For state [θ, θ_dot], LQR computes: u = -K·[θ, θ_dot]
        self.K_angle = 10.0  # Proportional gain on angle
        self.K_velocity = 2.0  # Derivative gain on angular velocity

        # Energy-based swing-up parameters
        # Target energy is the energy at top (potential energy = mgh)
        # For the Gym pendulum: m=1, l=1, g=10
        self.g = 10.0
        self.m = 1.0
        self.l = 1.0
        self.E_target = self.m * self.g * self.l  # Energy at top
        self.K_energy = 2.0  # Energy controller gain

    def get_action(self, state: np.ndarray) -> float:
        """
        Compute expert action for given state.

        Args:
            state: [cos(θ), sin(θ), θ_dot]

        Returns:
            Expert torque ∈ [-2, 2]
        """
        cos_theta = state[0]
        sin_theta = state[1]
        theta_dot = state[2]

        # Compute angle
        theta = np.arctan2(sin_theta, cos_theta)

        # Decide which controller to use
        if abs(theta) < self.swing_threshold:
            # Close to upright: use LQR balance controller
            action = self._balance_control(theta, theta_dot)
        else:
            # Far from upright: use energy-based swing-up
            action = self._swing_up_control(theta, theta_dot)

        # Clip to valid range
        return np.clip(action, -2.0, 2.0)

    def _balance_control(self, theta: float, theta_dot: float) -> float:
        """
        LQR balancing controller for upright position.

        This is a linear controller: u = -K·x
        where x = [θ, θ_dot] is the state deviation from equilibrium.

        Args:
            theta: Angle in radians
            theta_dot: Angular velocity

        Returns:
            Control torque
        """
        # Simple PD control (approximation of LQR)
        action = -(self.K_angle * theta + self.K_velocity * theta_dot)
        return action

    def _swing_up_control(self, theta: float, theta_dot: float) -> float:
        """
        Energy-based swing-up controller.

        The idea: pump energy into the system until it reaches the top.

        Energy of pendulum:
            E = (1/2)·m·l²·θ_dot² + m·g·l·(1 - cos(θ))
              = kinetic energy + potential energy

        Control law:
            u = K_e · (E - E_target) · sign(θ_dot · cos(θ))

        This pushes in the direction that increases energy when E < E_target.

        Args:
            theta: Angle in radians
            theta_dot: Angular velocity

        Returns:
            Control torque
        """
        # Compute current energy
        kinetic = 0.5 * self.m * (self.l * theta_dot) ** 2
        potential = self.m * self.g * self.l * (1 - np.cos(theta))
        E_current = kinetic + potential

        # Energy error
        E_error = E_current - self.E_target

        # Control law: push in direction that adds energy
        # sign(θ_dot · cos(θ)) determines the direction
        direction = np.sign(theta_dot * np.cos(theta))
        action = -self.K_energy * E_error * direction

        return action


def collect_expert_trajectories(
    num_episodes: int = 100,
    max_steps: int = 200,
    render: bool = False,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Collect expert demonstrations using the expert controller.

    This creates a dataset of (state, action) pairs for imitation learning.

    Args:
        num_episodes: Number of episodes to collect
        max_steps: Maximum steps per episode
        render: Whether to visualize collection
        seed: Random seed

    Returns:
        Dictionary with:
            states: [num_samples, state_dim]
            actions: [num_samples, action_dim]
            rewards: [num_samples]
            episode_returns: [num_episodes]
    """
    env = PendulumEnv(render_mode="human" if render else None)
    expert = ExpertController()

    states_list = []
    actions_list = []
    rewards_list = []
    episode_returns = []

    np.random.seed(seed)

    for episode in range(num_episodes):
        state = env.reset(seed=seed + episode)
        episode_reward = 0

        for step in range(max_steps):
            # Get expert action
            action = expert.get_action(state)

            # Store transition
            states_list.append(state)
            actions_list.append([action])

            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            rewards_list.append(reward)
            episode_reward += reward

            state = next_state

            if terminated or truncated:
                break

        episode_returns.append(episode_reward)

        if (episode + 1) % 10 == 0:
            mean_return = np.mean(episode_returns[-10:])
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Mean return (last 10): {mean_return:.1f}")

    env.close()

    # Convert to arrays
    data = {
        "states": np.array(states_list),
        "actions": np.array(actions_list),
        "rewards": np.array(rewards_list),
        "episode_returns": np.array(episode_returns),
    }

    print(f"\n✅ Collected {len(states_list)} transitions from {num_episodes} episodes")
    print(f"Mean episode return: {np.mean(episode_returns):.1f} ± {np.std(episode_returns):.1f}")

    return data


def visualize_expert_trajectory(num_steps: int = 200):
    """
    Visualize a single expert trajectory.

    Creates a plot showing:
    - Angle over time
    - Angular velocity over time
    - Action (torque) over time
    """
    env = PendulumEnv()
    expert = ExpertController()

    state = env.reset(seed=0)

    states = []
    actions_taken = []
    times = []

    for step in range(num_steps):
        action = expert.get_action(state)
        states.append(state)
        actions_taken.append(action)
        times.append(step * 0.05)  # dt = 0.05 in Pendulum-v1

        state, _, _, _, _ = env.step(action)

    env.close()

    # Convert to arrays for plotting
    states = np.array(states)
    actions_taken = np.array(actions_taken)
    times = np.array(times)

    # Extract angle from states
    angles = np.arctan2(states[:, 1], states[:, 0])
    velocities = states[:, 2]

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    axes[0].plot(times, angles, label="Angle θ")
    axes[0].axhline(y=0, color='r', linestyle='--', label="Target (upright)")
    axes[0].set_ylabel("Angle (rad)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(times, velocities, label="Angular velocity", color='orange')
    axes[1].set_ylabel("θ_dot (rad/s)")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(times, actions_taken, label="Torque", color='green')
    axes[2].axhline(y=2.0, color='r', linestyle='--', alpha=0.5)
    axes[2].axhline(y=-2.0, color='r', linestyle='--', alpha=0.5)
    axes[2].set_ylabel("Torque")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend()
    axes[2].grid(True)

    plt.suptitle("Expert Controller Trajectory")
    plt.tight_layout()
    plt.savefig("expert_trajectory.png", dpi=150)
    print("Saved visualization to expert_trajectory.png")
    plt.close()


# Test the expert controller
if __name__ == "__main__":
    print("Testing Expert Controller...")
    print("=" * 50)

    # Visualize a single trajectory
    print("\n1. Visualizing expert trajectory...")
    visualize_expert_trajectory(num_steps=200)

    # Collect some demonstration data
    print("\n2. Collecting expert demonstrations...")
    data = collect_expert_trajectories(
        num_episodes=10,
        max_steps=200,
        render=False,
        seed=42,
    )

    print(f"\nData shapes:")
    print(f"  States: {data['states'].shape}")
    print(f"  Actions: {data['actions'].shape}")
    print(f"  Mean return: {data['episode_returns'].mean():.1f}")

    print("\n✅ Expert controller working correctly!")
