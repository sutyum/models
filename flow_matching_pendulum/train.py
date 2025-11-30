"""
Training Script for Flow Matching Policy

This script:
1. Collects expert demonstrations from the expert controller
2. Trains a flow matching model to imitate the expert
3. Evaluates the learned policy
4. Saves checkpoints

Key concepts:
- Behavior Cloning: Learning from expert demonstrations
- Flow Matching: Treating action generation as a generative modeling problem
- Conditional Generation: Actions are conditioned on states
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Tuple
from tqdm import tqdm
import pickle
import os
from datetime import datetime

from flow_matching import FlowMatching, VelocityNet
from pendulum_env import PendulumEnv, collect_expert_trajectories


class FlowMatchingPolicy:
    """
    A policy that uses flow matching to generate actions.

    This wraps the VelocityNet and provides a simple interface:
        action = policy(state)
    """

    def __init__(
        self,
        model: VelocityNet,
        num_sampling_steps: int = 10,
        sampling_method: str = "euler",
        device: torch.device = None,
    ):
        """
        Initialize the policy.

        Args:
            model: Trained velocity network
            num_sampling_steps: Number of ODE integration steps
            sampling_method: "euler" or "heun"
            device: Device to run on
        """
        self.model = model
        self.num_sampling_steps = num_sampling_steps
        self.sampling_method = sampling_method
        self.device = device if device is not None else torch.device("cpu")
        self.model.to(self.device)

    def __call__(self, state: np.ndarray) -> np.ndarray:
        """
        Generate action for given state.

        Args:
            state: Current state [state_dim]

        Returns:
            Action [action_dim]
        """
        # Convert to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)

        # Sample action using flow matching
        with torch.no_grad():
            action_tensor = FlowMatching.sample_action(
                self.model,
                state_tensor,
                num_steps=self.num_sampling_steps,
                method=self.sampling_method,
                device=self.device,
            )

        # Convert back to numpy
        return action_tensor.cpu().numpy()


class ExpertDataset(Dataset):
    """PyTorch dataset for expert demonstrations."""

    def __init__(self, states: np.ndarray, actions: np.ndarray):
        self.states = torch.tensor(states, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


def train_flow_matching_policy(
    num_expert_episodes: int = 100,
    num_epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    hidden_dim: int = 256,
    num_layers: int = 3,
    seed: int = 42,
    eval_every: int = 5,
    save_dir: str = "checkpoints",
    device: str = "cpu",
) -> Tuple[VelocityNet, Dict]:
    """
    Train a flow matching policy from expert demonstrations.

    Args:
        num_expert_episodes: Number of expert episodes to collect
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for Adam optimizer
        hidden_dim: Hidden dimension for velocity network
        num_layers: Number of layers in velocity network
        seed: Random seed
        eval_every: Evaluate policy every N epochs
        save_dir: Directory to save checkpoints
        device: Device to train on ("cpu" or "cuda")

    Returns:
        Trained model and training history
    """
    print("=" * 70)
    print("FLOW MATCHING FOR INVERTED PENDULUM")
    print("=" * 70)

    # Set device
    device = torch.device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Step 1: Collect expert demonstrations
    print("\n📊 Step 1: Collecting Expert Demonstrations")
    print("-" * 70)
    expert_data = collect_expert_trajectories(
        num_episodes=num_expert_episodes,
        max_steps=200,
        render=False,
        seed=seed,
    )

    # Step 2: Initialize model
    print("\n🧠 Step 2: Initializing Flow Matching Model")
    print("-" * 70)
    state_dim = 3  # [cos(θ), sin(θ), θ_dot]
    action_dim = 1  # [torque]

    model = VelocityNet(
        action_dim=action_dim,
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)

    print(f"Model architecture:")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Num layers: {num_layers}")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {num_params:,}")

    # Create dataset and dataloader
    dataset = ExpertDataset(expert_data["states"], expert_data["actions"])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Step 3: Setup optimizer
    print("\n⚙️  Step 3: Setting up Optimizer")
    print("-" * 70)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Optimizer: Adam")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Num epochs: {num_epochs}")

    # Step 4: Training loop
    print("\n🚀 Step 4: Training")
    print("-" * 70)

    history = {
        "train_loss": [],
        "eval_returns": [],
        "epochs": [],
    }

    best_return = -float("inf")

    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_losses = []

        for states_batch, actions_batch in tqdm(
            dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False
        ):
            states_batch = states_batch.to(device)
            actions_batch = actions_batch.to(device)

            # Compute loss
            loss = FlowMatching.compute_loss(model, states_batch, actions_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        mean_loss = np.mean(epoch_losses)
        history["train_loss"].append(mean_loss)
        history["epochs"].append(epoch)

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {mean_loss:.4f}")

        # Evaluation
        if (epoch + 1) % eval_every == 0 or epoch == num_epochs - 1:
            print(f"  Evaluating policy...")
            model.eval()
            eval_return = evaluate_policy(
                model, num_episodes=5, num_sampling_steps=10, device=device
            )
            history["eval_returns"].append(eval_return)
            print(f"  Eval return: {eval_return:.1f}")

            # Save best model
            if eval_return > best_return:
                best_return = eval_return
                save_path = os.path.join(save_dir, f"best_model_{timestamp}.pt")
                torch.save(model.state_dict(), save_path)
                print(f"  💾 Saved new best model (return: {eval_return:.1f})")

    # Save final model
    final_path = os.path.join(save_dir, f"final_model_{timestamp}.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\n💾 Saved final model to {final_path}")

    # Save history
    history_path = os.path.join(save_dir, f"history_{timestamp}.pkl")
    with open(history_path, "wb") as f:
        pickle.dump(history, f)
    print(f"💾 Saved training history to {history_path}")

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Best eval return: {best_return:.1f}")
    print(f"Final loss: {history['train_loss'][-1]:.4f}")

    return model, history


def evaluate_policy(
    model: VelocityNet,
    num_episodes: int = 10,
    num_sampling_steps: int = 10,
    device: torch.device = None,
    render: bool = False,
) -> float:
    """
    Evaluate the learned policy.

    Args:
        model: Trained velocity network
        num_episodes: Number of episodes to evaluate
        num_sampling_steps: Flow sampling steps
        device: Device to run on
        render: Whether to visualize

    Returns:
        Mean episode return
    """
    if device is None:
        device = torch.device("cpu")

    env = PendulumEnv(render_mode="human" if render else None)
    policy = FlowMatchingPolicy(model, num_sampling_steps=num_sampling_steps, device=device)

    returns = []

    for episode in range(num_episodes):
        state = env.reset(seed=episode)
        episode_return = 0

        for step in range(200):
            # Generate action using flow matching
            action = policy(state)

            # Take step
            state, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward

            if terminated or truncated:
                break

        returns.append(episode_return)

    env.close()

    mean_return = np.mean(returns)
    return mean_return


if __name__ == "__main__":
    # Train the policy
    model, history = train_flow_matching_policy(
        num_expert_episodes=100,
        num_epochs=50,
        batch_size=256,
        learning_rate=3e-4,
        hidden_dim=256,
        num_layers=3,
        seed=42,
        eval_every=5,
        device="cpu",
    )

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    final_return = evaluate_policy(
        model,
        num_episodes=20,
        num_sampling_steps=20,  # Use more steps for final eval
        device=torch.device("cpu"),
        render=False,
    )

    print(f"Final policy return (20 episodes): {final_return:.1f}")
    print("\n✅ All done! Your flow matching policy is ready.")
    print("\nNext steps:")
    print("  1. Run eval.py to visualize the learned policy")
    print("  2. Experiment with different architectures and hyperparameters")
    print("  3. Try scaling to humanoid locomotion!")
