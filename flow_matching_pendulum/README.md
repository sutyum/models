# Flow Matching for Inverted Pendulum Control

An educational implementation of **Flow Matching** applied to reinforcement learning control problems, starting with the classic inverted pendulum and designed to scale to humanoid locomotion.

## 🎯 What is Flow Matching?

Flow Matching is a generative modeling technique that learns to transform a simple source distribution (e.g., Gaussian noise) into a complex target distribution by learning a continuous-time flow.

### Key Concepts

**1. Continuous Normalizing Flows (CNFs)**
- Think of it as learning a vector field that "pushes" samples from noise to data
- At each point in space-time, the model predicts which direction to move
- Starting from random noise at t=0, we follow the flow to get a sample at t=1

**2. Flow Matching Objective**
Unlike diffusion models that add/remove noise, flow matching directly learns the velocity field:

```
Flow ODE: dx/dt = v_θ(x, t)
```

Where:
- `x` is the state (position in data space)
- `t` is time ∈ [0, 1]
- `v_θ` is our learned velocity field (neural network)

**3. Training with Optimal Transport**
We use the *conditional flow matching* objective:
```
L(θ) = E[||v_θ(x_t, t) - (x_1 - x_0)||²]
```

Where:
- `x_0 ~ N(0, I)` (source: Gaussian noise)
- `x_1 ~ p_data` (target: expert demonstrations)
- `x_t = (1-t)x_0 + t·x_1` (linear interpolation)
- The optimal velocity is simply `x_1 - x_0`!

This is beautifully simple: the model learns to predict the direction from noise to data.

## 🤸 Application to RL: Why Flow Matching for Control?

Traditional RL methods (PPO, SAC, etc.) optimize for cumulative reward. Flow matching offers an alternative:

**Behavior Cloning as Generative Modeling**
- Expert demonstrations define a distribution over state-action pairs
- Flow matching learns to generate actions conditioned on states
- Benefits:
  - Can model multimodal action distributions
  - Smooth interpolation in action space
  - Naturally handles continuous control
  - Can be combined with diffusion-style iterative refinement

**For the Inverted Pendulum:**
- State: `[cos(θ), sin(θ), θ_dot]` (angle and angular velocity)
- Action: `[torque]` (continuous control)
- Goal: Learn to map states → actions from expert trajectories

## 🏗️ Architecture Overview

```
State → [Encoder] → Embedding
                        ↓
Noise z_0 ────────────→ [Flow Network] → Refined Action
Time t ────────────────→      ↑
                              │
                    Predicts velocity v_θ(z_t, t, state)
```

### Flow Network Design
- **Input**: Current noisy action `z_t`, time `t`, state embedding
- **Output**: Velocity vector (direction to move in action space)
- **Architecture**: MLP with time and state conditioning

## 📚 Scaling to Humanoids

This implementation is designed with humanoid locomotion in mind:

1. **State Representation**: Easy to extend from 3D pendulum state to high-dimensional humanoid state (joint angles, velocities, orientation, etc.)

2. **Action Dimension**: Pendulum has 1 action (torque), humanoids have ~20+ (joint torques). Flow matching scales naturally to high dimensions.

3. **Multimodal Behaviors**: Humanoids may have multiple valid gaits (walking, running, jumping). Flow matching can capture these modes.

4. **Velocity Tracking**: The pendulum teaches you to match reference velocities. For humanoids, you'll track desired walking velocities - same concept, higher dimensions!

## 🚀 Quick Start

```bash
# Install dependencies
uv sync

# Train a flow matching policy from expert data
python train.py --episodes 1000 --flow-steps 10

# Evaluate the learned policy
python eval.py --checkpoint checkpoints/best.eqx

# Visualize the flow field
python visualize.py
```

## 📖 Code Structure

- `flow_matching.py`: Core flow matching implementation
- `policy.py`: Flow-based policy network
- `pendulum_env.py`: Environment wrapper and expert controller
- `train.py`: Training loop with expert data collection
- `eval.py`: Evaluation and visualization
- `utils.py`: Helper functions

## 🧠 Key Insights for Learning

1. **Flow Matching vs Diffusion**: Flow matching uses straight paths (OT), diffusion uses curved paths. Flow matching is often simpler to train and faster to sample.

2. **Conditional Generation**: We condition the flow on the state, making this a *conditional flow matching* problem.

3. **Trade-off**: More flow steps → better quality but slower. Start with 10-20 steps.

4. **Data Efficiency**: Flow matching can be data-efficient since it directly learns from demonstrations without rewards.

## 📊 What You'll Learn

- [x] Flow matching theory and implementation
- [x] Conditional generation for RL
- [x] Neural ODEs and continuous-time modeling
- [x] Behavior cloning with generative models
- [x] Foundation for scaling to complex robots

## 🔗 References

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) (Lipman et al., 2023)
- [Diffusion Models for Reinforcement Learning](https://arxiv.org/abs/2205.09991) (Diffuser)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (DDPM)

---

**Next Steps**: Once you master this, you can apply the same techniques to humanoid locomotion using environments like `dm_control humanoid` or `Isaac Gym`!
