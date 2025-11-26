# Flow Matching Quick Start Guide

## 🎯 What You Have

A complete, educational implementation of **Flow Matching** for learning control policies! This teaches you:
- How to use generative modeling for RL
- Flow matching theory and practice
- Foundation for humanoid locomotion control

## 🚀 Get Started in 5 Minutes

### 1. See the Expert in Action
```bash
cd flow_matching_pendulum
uv run python main.py demo
```
This creates `expert_trajectory.png` showing the energy-based swing-up controller.

### 2. Run a Quick Test
```bash
uv run python main.py test
```
Trains a tiny model to verify everything works (~2 minutes).

### 3. Train a Real Policy
```bash
uv run python main.py train --episodes 100 --epochs 50
```
Full training run (~10-15 minutes). Creates checkpoints in `checkpoints/`.

### 4. Evaluate and Visualize
```bash
uv run python eval.py --checkpoint checkpoints/best_model_*.eqx --visualize
```
Creates beautiful visualizations comparing your policy to the expert!

## 📊 What the Visualizations Show

- **expert_trajectory.png**: Expert controller behavior (swing-up + balance)
- **rollout_comparison.png**: Your policy vs expert side-by-side
- **flow_field.png**: The learned velocity field v_θ(z, t, state)
- **denoising_process.png**: How noise → action transformation works

## 🎓 Learning Path

1. **Read the Theory** (`README.md`): Understand flow matching basics
2. **Study the Code** (`flow_matching.py`): See implementation with detailed comments
3. **Follow the Tutorial** (`TUTORIAL.md`): Deep dive into theory and practice
4. **Experiment**: Try different hyperparameters, visualizations, etc.
5. **Scale Up**: Apply to humanoid locomotion!

## 🔧 Key Files

```
flow_matching_pendulum/
├── flow_matching.py      # Core: VelocityNet, loss, sampling
├── pendulum_env.py       # Environment + expert controller
├── train.py              # Training loop
├── eval.py               # Evaluation + visualizations
├── main.py               # CLI interface
├── README.md             # Project overview
├── TUTORIAL.md           # Comprehensive tutorial
└── QUICKSTART.md         # This file!
```

## 🎮 Common Commands

```bash
# Demo
uv run python main.py demo

# Quick test
uv run python main.py test

# Full training with custom params
uv run python main.py train \
    --episodes 200 \
    --epochs 100 \
    --batch-size 256 \
    --lr 0.0003 \
    --hidden-dim 256

# Evaluate
uv run python eval.py \
    --checkpoint checkpoints/best_model_*.eqx \
    --num-episodes 20 \
    --num-steps 20 \
    --visualize

# Test individual modules
uv run python flow_matching.py    # Test core implementation
uv run python pendulum_env.py     # Test environment + expert
```

## 💡 Pro Tips

1. **Start small**: Use `main.py test` first to verify everything works
2. **More data helps**: Try 100-200 expert episodes for best results
3. **Sampling steps trade-off**: 10 steps is fast, 20 is better quality
4. **Watch the loss**: Should decrease from ~3.0 to ~1.0 or lower
5. **Compare with expert**: Your policy should get within 80-90% of expert performance

## 🐛 Troubleshooting

**Model not learning?**
- Increase epochs (try 100)
- Collect more expert data (try 200 episodes)
- Increase model size (hidden_dim=512, num_layers=4)

**Actions too noisy?**
- Use more sampling steps (20-50)
- Use Heun integration instead of Euler (better but slower)

**Want better performance?**
- The expert is energy-based + LQR, achieving ~-200 to -500 return
- Flow matching should get close with enough data and training
- Try combining with RL fine-tuning for best results

## 🚀 Next: Humanoids!

Once you're comfortable with the pendulum:

1. **Read TUTORIAL.md Part 4**: Scaling to humanoids
2. **Choose environment**: dm_control humanoid, Isaac Gym, or MuJoCo
3. **Adapt the code**:
   - Change state_dim (3 → 50-200)
   - Change action_dim (1 → 20-30)
   - Add velocity conditioning
4. **Collect expert data**: RL policy or motion capture
5. **Train**: Same algorithm, just bigger!

## 📚 Learning Resources

- **Flow Matching Paper**: https://arxiv.org/abs/2210.02747
- **Diffusion Policy**: Similar idea for robotics
- **Score Matching**: Related generative approach

## ✅ Success Criteria

You're ready for humanoids when you can:
- [ ] Explain flow matching in your own words
- [ ] Understand the VelocityNet architecture
- [ ] Train a policy that gets >80% of expert performance
- [ ] Create and interpret visualizations
- [ ] Modify hyperparameters and see effects

## 🎉 Have Fun!

Flow matching is powerful and elegant. Enjoy learning, experimenting, and scaling to complex robots!

Questions? Check TUTORIAL.md or open an issue on GitHub.
