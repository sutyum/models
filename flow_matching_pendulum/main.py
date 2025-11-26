"""
Main entry point for Flow Matching Pendulum project.

This provides a simple CLI to run different parts of the project.
"""

import argparse
from train import train_flow_matching_policy
from pendulum_env import collect_expert_trajectories, visualize_expert_trajectory


def main():
    parser = argparse.ArgumentParser(
        description="Flow Matching for Inverted Pendulum",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize expert controller
  python main.py demo

  # Collect expert data only
  python main.py collect --episodes 100

  # Train a policy
  python main.py train --episodes 100 --epochs 50

  # Quick test (small training run)
  python main.py test
        """
    )

    parser.add_argument(
        "mode",
        choices=["demo", "collect", "train", "test"],
        help="What to run: demo (visualize expert), collect (gather data), train (full training), test (quick test)"
    )

    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of expert episodes to collect")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="Hidden dimension for network")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    if args.mode == "demo":
        print("🎬 Visualizing Expert Controller")
        print("=" * 70)
        visualize_expert_trajectory(num_steps=200)
        print("\n✅ Check expert_trajectory.png to see the expert controller in action!")

    elif args.mode == "collect":
        print("📊 Collecting Expert Demonstrations")
        print("=" * 70)
        data = collect_expert_trajectories(
            num_episodes=args.episodes,
            max_steps=200,
            render=False,
            seed=args.seed,
        )
        print(f"\n✅ Collected {len(data['states'])} transitions")

    elif args.mode == "train":
        print("🚀 Training Flow Matching Policy")
        print("=" * 70)
        model, history = train_flow_matching_policy(
            num_expert_episodes=args.episodes,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            hidden_dim=args.hidden_dim,
            num_layers=3,
            seed=args.seed,
            eval_every=max(1, args.epochs // 10),
        )
        print("\n✅ Training complete! Check the checkpoints/ directory for saved models.")
        print("   Run eval.py to visualize the learned policy.")

    elif args.mode == "test":
        print("🧪 Quick Test (Small Training Run)")
        print("=" * 70)
        print("This will run a small training to verify everything works.")
        print()
        model, history = train_flow_matching_policy(
            num_expert_episodes=20,   # Small dataset
            num_epochs=5,             # Few epochs
            batch_size=64,            # Small batches
            learning_rate=3e-4,
            hidden_dim=128,           # Smaller network
            num_layers=2,
            seed=args.seed,
            eval_every=2,
        )
        print("\n✅ Test complete! Everything is working correctly.")
        print("   For real training, use: python main.py train")


if __name__ == "__main__":
    main()
