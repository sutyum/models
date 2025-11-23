"""
Evaluation and Visualization for Flow Matching Policy

This script provides tools to:
1. Evaluate a trained policy
2. Visualize rollouts
3. Compare with expert controller
4. Analyze the flow field
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from flow_matching import VelocityNet, FlowMatching
from pendulum_env import PendulumEnv, ExpertController
from train import FlowMatchingPolicy


def load_model(checkpoint_path: str, action_dim: int = 1, state_dim: int = 3, device: str = "cpu") -> VelocityNet:
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        action_dim: Action dimension
        state_dim: State dimension
        device: Device to load model on

    Returns:
        Loaded VelocityNet model
    """
    # Create a model with the same architecture
    model = VelocityNet(
        action_dim=action_dim,
        state_dim=state_dim,
        hidden_dim=256,  # Should match training config
        num_layers=3,
    )

    # Load the trained weights
    device_obj = torch.device(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device_obj))
    model.to(device_obj)
    model.eval()

    return model


def evaluate_and_compare(
    model: VelocityNet,
    num_episodes: int = 10,
    num_sampling_steps: int = 20,
    seed: int = 42,
    device: str = "cpu",
):
    """
    Evaluate flow matching policy and compare with expert.

    Args:
        model: Trained velocity network
        num_episodes: Number of episodes to evaluate
        num_sampling_steps: Flow sampling steps
        seed: Random seed
        device: Device to run on
    """
    device_obj = torch.device(device)
    env = PendulumEnv()
    policy = FlowMatchingPolicy(model, num_sampling_steps=num_sampling_steps, device=device_obj)
    expert = ExpertController()

    np.random.seed(seed)

    flow_returns = []
    expert_returns = []

    print(f"Evaluating over {num_episodes} episodes...")
    print("=" * 70)

    for episode in range(num_episodes):
        # Evaluate flow matching policy
        state = env.reset(seed=seed + episode)
        flow_return = 0

        for step in range(200):
            action = policy(state)
            state, reward, terminated, truncated, _ = env.step(action)
            flow_return += reward

            if terminated or truncated:
                break

        flow_returns.append(flow_return)

        # Evaluate expert policy
        state = env.reset(seed=seed + episode)
        expert_return = 0

        for step in range(200):
            action = expert.get_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            expert_return += reward

            if terminated or truncated:
                break

        expert_returns.append(expert_return)

        print(f"Episode {episode + 1}: "
              f"Flow={flow_return:.1f}, Expert={expert_return:.1f}")

    env.close()

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Flow Matching Policy:")
    print(f"  Mean return: {np.mean(flow_returns):.1f} ± {np.std(flow_returns):.1f}")
    print(f"  Min/Max: {np.min(flow_returns):.1f} / {np.max(flow_returns):.1f}")
    print(f"\nExpert Controller:")
    print(f"  Mean return: {np.mean(expert_returns):.1f} ± {np.std(expert_returns):.1f}")
    print(f"  Min/Max: {np.min(expert_returns):.1f} / {np.max(expert_returns):.1f}")

    success_rate = np.mean(np.array(flow_returns) > -200)  # Reasonable threshold
    print(f"\nSuccess rate (return > -200): {success_rate * 100:.1f}%")


def visualize_rollout(
    model: VelocityNet,
    num_sampling_steps: int = 20,
    num_steps: int = 200,
    seed: int = 0,
    save_path: str = "rollout_comparison.png",
    device: str = "cpu",
):
    """
    Visualize a rollout comparing flow policy and expert.

    Args:
        model: Trained velocity network
        num_sampling_steps: Flow sampling steps
        num_steps: Number of steps to simulate
        seed: Random seed
        save_path: Path to save the plot
        device: Device to run on
    """
    device_obj = torch.device(device)
    env = PendulumEnv()
    policy = FlowMatchingPolicy(model, num_sampling_steps=num_sampling_steps, device=device_obj)
    expert = ExpertController()

    # Collect flow policy trajectory
    state = env.reset(seed=seed)
    flow_states = []
    flow_actions = []

    for step in range(num_steps):
        action = policy(state)
        flow_states.append(state)
        flow_actions.append(action)
        state, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            break

    # Collect expert trajectory
    state = env.reset(seed=seed)
    expert_states = []
    expert_actions = []

    for step in range(num_steps):
        action = expert.get_action(state)
        expert_states.append(state)
        expert_actions.append(action)
        state, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            break

    env.close()

    # Convert to arrays
    flow_states = np.array(flow_states)
    flow_actions = np.array(flow_actions)
    expert_states = np.array(expert_states)
    expert_actions = np.array(expert_actions)

    # Extract angles
    flow_angles = np.arctan2(flow_states[:, 1], flow_states[:, 0])
    expert_angles = np.arctan2(expert_states[:, 1], expert_states[:, 0])
    flow_velocities = flow_states[:, 2]
    expert_velocities = expert_states[:, 2]

    times = np.arange(len(flow_angles)) * 0.05

    # Create plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    # Angle
    axes[0].plot(times, flow_angles, label="Flow Matching", linewidth=2)
    axes[0].plot(times, expert_angles, label="Expert", linestyle='--', linewidth=2)
    axes[0].axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    axes[0].set_ylabel("Angle θ (rad)", fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Trajectory Comparison: Flow Matching vs Expert", fontsize=14)

    # Angular velocity
    axes[1].plot(times, flow_velocities, label="Flow Matching", linewidth=2)
    axes[1].plot(times, expert_velocities, label="Expert", linestyle='--', linewidth=2)
    axes[1].axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    axes[1].set_ylabel("θ_dot (rad/s)", fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # Actions
    axes[2].plot(times, flow_actions, label="Flow Matching", linewidth=2)
    axes[2].plot(times, expert_actions, label="Expert", linestyle='--', linewidth=2)
    axes[2].axhline(y=2.0, color='red', linestyle=':', alpha=0.5)
    axes[2].axhline(y=-2.0, color='red', linestyle=':', alpha=0.5)
    axes[2].set_ylabel("Torque", fontsize=12)
    axes[2].set_xlabel("Time (s)", fontsize=12)
    axes[2].legend(fontsize=11)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved rollout comparison to {save_path}")
    plt.close()


def visualize_flow_field(
    model: VelocityNet,
    t: float = 0.5,
    state: np.ndarray = None,
    save_path: str = "flow_field.png",
    device: str = "cpu",
):
    """
    Visualize the learned flow field at a specific time.

    Args:
        model: Trained velocity network
        t: Time parameter to visualize
        state: State to condition on (default: near-upright)
        save_path: Path to save the plot
        device: Device to run on
    """
    device_obj = torch.device(device)
    model.to(device_obj)

    if state is None:
        # Default: pendulum nearly upright
        state = np.array([1.0, 0.0, 0.0])  # [cos(0), sin(0), 0]

    state_tensor = torch.tensor(state, dtype=torch.float32, device=device_obj)

    # Create grid in action space
    # For 1D action (pendulum), we'll visualize action vs time
    if model.action_dim == 1:
        # Grid: action on x-axis, time on y-axis
        actions = np.linspace(-3, 3, 50)
        times = np.linspace(0, 1, 50)
        A, T = np.meshgrid(actions, times)

        # Compute velocity at each grid point
        velocities = np.zeros_like(A)

        with torch.no_grad():
            for i in range(len(times)):
                for j in range(len(actions)):
                    z = torch.tensor([[actions[j]]], dtype=torch.float32, device=device_obj)
                    t_val = torch.tensor([times[i]], dtype=torch.float32, device=device_obj)
                    v = model(z, t_val, state_tensor.unsqueeze(0))
                    velocities[i, j] = v[0, 0].item()

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        contour = ax.contourf(A, T, velocities, levels=20, cmap='RdBu_r')
        plt.colorbar(contour, label='Velocity')

        # Add flow lines
        ax.contour(A, T, velocities, levels=10, colors='black', alpha=0.3, linewidths=0.5)

        ax.set_xlabel('Action (noisy)', fontsize=12)
        ax.set_ylabel('Time t', fontsize=12)
        ax.set_title(f'Flow Field v_θ(z, t) at state=[{state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f}]',
                     fontsize=14)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved flow field visualization to {save_path}")
        plt.close()


def visualize_denoising_process(
    model: VelocityNet,
    state: np.ndarray,
    num_steps: int = 10,
    save_path: str = "denoising_process.png",
    device: str = "cpu",
):
    """
    Visualize how noise is transformed into an action.

    Args:
        model: Trained velocity network
        state: State to condition on
        num_steps: Number of denoising steps
        save_path: Path to save the plot
        device: Device to run on
    """
    device_obj = torch.device(device)
    model.to(device_obj)

    state_tensor = torch.tensor(state, dtype=torch.float32, device=device_obj)

    # Sample initial noise
    z = torch.randn(model.action_dim, device=device_obj)

    # Track trajectory
    trajectory = [z[0].item()]
    times = [0.0]

    dt = 1.0 / num_steps

    with torch.no_grad():
        for step in range(num_steps):
            t = torch.tensor([step * dt], dtype=torch.float32, device=device_obj)
            velocity = model(z.unsqueeze(0), t, state_tensor.unsqueeze(0)).squeeze(0)
            z = z + dt * velocity
            trajectory.append(z[0].item())
            times.append((step + 1) * dt)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(times, trajectory, 'o-', linewidth=2, markersize=8, label='Flow trajectory')
    ax.axhline(y=trajectory[-1], color='green', linestyle='--', label=f'Final action: {trajectory[-1]:.3f}')
    ax.axhline(y=trajectory[0], color='red', linestyle='--', alpha=0.5, label=f'Initial noise: {trajectory[0]:.3f}')

    ax.set_xlabel('Time t', fontsize=12)
    ax.set_ylabel('Action value', fontsize=12)
    ax.set_title('Flow Matching: Transforming Noise → Action', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved denoising process to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Flow Matching Policy")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--num-episodes", type=int, default=10,
                        help="Number of episodes to evaluate")
    parser.add_argument("--num-steps", type=int, default=20,
                        help="Number of flow sampling steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--visualize", action="store_true",
                        help="Create visualizations")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run on (cpu or cuda)")

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device=args.device)
    print("✅ Model loaded successfully")

    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)
    evaluate_and_compare(
        model,
        num_episodes=args.num_episodes,
        num_sampling_steps=args.num_steps,
        seed=args.seed,
        device=args.device,
    )

    # Visualizations
    if args.visualize:
        print("\n" + "=" * 70)
        print("CREATING VISUALIZATIONS")
        print("=" * 70)

        # Rollout comparison
        visualize_rollout(model, num_sampling_steps=args.num_steps, seed=args.seed, device=args.device)

        # Flow field
        visualize_flow_field(model, t=0.5, device=args.device)

        # Denoising process
        test_state = np.array([1.0, 0.0, 0.0])  # Upright
        visualize_denoising_process(model, test_state, num_steps=args.num_steps, device=args.device)

        print("\n✅ All visualizations created!")


if __name__ == "__main__":
    main()
