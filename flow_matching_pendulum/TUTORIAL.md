# Flow Matching Tutorial: From Theory to Practice

This tutorial will guide you through understanding and using flow matching for control tasks, starting with the inverted pendulum and preparing you for humanoid locomotion.

## 🎓 Part 1: Understanding Flow Matching

### What Problem Are We Solving?

In reinforcement learning, we often want to learn a policy: `π(a|s)` - the probability of taking action `a` given state `s`.

Traditional approaches:
- **Policy Gradient (PPO, etc.)**: Maximize expected reward through gradient descent
- **Q-Learning (DQN, SAC)**: Learn value functions and derive policies
- **Behavior Cloning**: Directly imitate expert demonstrations

Flow matching takes a generative modeling approach to behavior cloning.

### The Flow Matching Idea

Think of flow matching as learning to "denoise" actions:

1. **Start**: Random noise `z₀ ~ N(0, I)`
2. **Flow**: Continuously transform noise → action via ODE: `dz/dt = v_θ(z, t, state)`
3. **End**: Clean action `a` that the expert would take

The magic: We train `v_θ` (our neural network) to predict the velocity field that transforms noise into expert actions.

### Why This Works

The **Conditional Flow Matching** loss is surprisingly simple:

```
L(θ) = E[||v_θ(z_t, t, s) - (x_expert - z_0)||²]
```

Where:
- `z_t = (1-t)·z_0 + t·x_expert` (linear interpolation)
- `x_expert - z_0` is the optimal velocity (just the direction from noise to data!)

This means we're training the network to predict: "If I'm at point `z_t` at time `t`, which direction should I move to reach the expert action?"

## 🔬 Part 2: Code Walkthrough

### Architecture Overview

```
State: [cos(θ), sin(θ), θ_dot]
   ↓
Noise z₀ ~ N(0,I) ─────→ VelocityNet(z, t, state) ─→ velocity
                              ↓
                         ODE Integration (Euler/Heun)
                              ↓
                         Action: [torque]
```

### Key Components

#### 1. VelocityNet (flow_matching.py)

The core neural network that predicts velocities:

```python
class VelocityNet:
    def __call__(self, z, t, state):
        # z: current noisy action
        # t: time ∈ [0, 1]
        # state: environment state

        # Create time embedding (sinusoidal, like in Transformers)
        t_embed = time_embedding(t)

        # Concatenate all inputs
        x = concat([z, t_embed, state])

        # Forward through MLP
        velocity = MLP(x)

        return velocity
```

**Why time embedding?** The model needs to know "when" in the denoising process we are. Different times require different velocities. We use sinusoidal embeddings (like positional encoding) to give the model a rich representation of time.

#### 2. Flow Matching Loss (flow_matching.py)

```python
def compute_loss(model, states, actions):
    # Sample random time for each example
    t = uniform(0, 1)

    # Create flow path: noise → expert action
    z_0 = random_normal()
    x_1 = expert_action
    z_t = (1-t) * z_0 + t * x_1

    # Optimal velocity is constant along the path!
    v_optimal = x_1 - z_0

    # Predict velocity with our model
    v_pred = model(z_t, t, state)

    # Mean squared error
    loss = ||v_pred - v_optimal||²

    return loss
```

This is the entire training objective! Much simpler than diffusion models.

#### 3. Sampling Actions (flow_matching.py)

```python
def sample_action(model, state, num_steps=20):
    # Start from random noise
    z = random_normal()

    # Integrate the ODE from t=0 to t=1
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t = step * dt

        # Get velocity from model
        velocity = model(z, t, state)

        # Take a step (Euler method)
        z = z + dt * velocity

    # z is now our action!
    return z
```

This is equivalent to solving the ODE: `dz/dt = v_θ(z, t, state)` from `t=0` to `t=1`.

### Training Process

1. **Collect Expert Data** (pendulum_env.py):
   - Run expert controller (energy-based swing-up + LQR balance)
   - Save (state, action) pairs

2. **Train Flow Model** (train.py):
   - For each batch of (state, action):
     - Sample random time `t`
     - Create flow path `z_t = (1-t)·noise + t·action`
     - Compute optimal velocity `v* = action - noise`
     - Predict velocity `v_pred = model(z_t, t, state)`
     - Update model to minimize `||v_pred - v*||²`

3. **Evaluate** (eval.py):
   - For each state, sample action by solving flow ODE
   - Compare with expert performance

## 🎮 Part 3: Hands-On Usage

### Quick Start

```bash
# Install dependencies
cd flow_matching_pendulum
uv sync

# See expert controller in action
uv run python main.py demo

# Run a quick test (small training run)
uv run python main.py test

# Full training (100 episodes, 50 epochs)
uv run python main.py train --episodes 100 --epochs 50

# Evaluate trained model with visualizations
uv run python eval.py --checkpoint checkpoints/best_model_*.eqx --visualize
```

### Understanding the Results

**Expert controller** achieves around `-200 to -500` mean return on pendulum. Your flow matching policy should get close to this!

**What the visualizations show:**
- `expert_trajectory.png`: How the expert controller swings up and balances
- `rollout_comparison.png`: Your policy vs expert side-by-side
- `flow_field.png`: The learned velocity field in action space
- `denoising_process.png`: How noise transforms into an action

### Hyperparameter Tuning

Key hyperparameters to experiment with:

1. **Number of sampling steps** (10-50):
   - More steps → better quality, slower inference
   - Start with 10 for training, 20 for evaluation

2. **Network architecture**:
   - `hidden_dim`: 128-512
   - `num_layers`: 2-4
   - Bigger networks learn faster but may overfit

3. **Training**:
   - `learning_rate`: 1e-4 to 3e-4 works well
   - `batch_size`: 64-256
   - `num_epochs`: 20-100 depending on dataset size

4. **Data collection**:
   - More expert episodes → better performance
   - 50-200 episodes is usually sufficient

## 🚀 Part 4: Scaling to Humanoids

### What Changes for Humanoid Locomotion?

1. **State Dimension**:
   - Pendulum: 3D `[cos(θ), sin(θ), θ_dot]`
   - Humanoid: 50-200D (all joint angles, velocities, body orientation, etc.)

2. **Action Dimension**:
   - Pendulum: 1D `[torque]`
   - Humanoid: 20-30D (torque for each actuated joint)

3. **Complexity**:
   - Pendulum: Single swing-up motion
   - Humanoid: Complex walking gaits, balance, coordination

### How Flow Matching Helps

✅ **Scales naturally to high dimensions**: The flow matching loss doesn't change! Still just `||v_pred - v_optimal||²`

✅ **Captures multimodal behaviors**: Humanoids can walk/run/jump - flow matching can learn all modes

✅ **Smooth interpolation**: Actions are generated via continuous flow, ensuring smooth motor control

✅ **Velocity tracking**: Once you learn to track velocities on pendulum, same concept applies to humanoid walking at different speeds

### Next Steps for Humanoids

1. **Environment**: Use `dm_control humanoid`, `Isaac Gym`, or `MuJoCo humanoid`

2. **Expert Data**:
   - Option 1: Train RL policy first (PPO), then distill with flow matching
   - Option 2: Use motion capture data (harder to set up)
   - Option 3: Trajectory optimization (CasADi, Drake)

3. **Architecture Changes**:
   - Increase `hidden_dim` to 512-1024
   - Add more layers (3-5)
   - Consider attention mechanisms for long-range dependencies

4. **Conditional Training**:
   - Condition on desired velocity: `v_θ(z, t, state, velocity_target)`
   - Enables steering the robot at different speeds!

### Example: Velocity Tracking

```python
# In your velocity network, add velocity_target as input
class VelocityNet:
    def __call__(self, z, t, state, velocity_target):
        # Now the model learns: "given I want to walk at
        # velocity_target, what action should I take in state?"
        x = concat([z, time_embed(t), state, velocity_target])
        return MLP(x)

# At test time, specify desired velocity
desired_velocity = [1.5, 0.0]  # 1.5 m/s forward, 0 m/s sideways
action = policy(state, velocity_target=desired_velocity)
```

## 🎯 Part 5: Exercises & Experiments

### Beginner

1. **Visualize the flow field**: Run with different states (upright, hanging, halfway)
2. **Ablation study**: Try 5, 10, 20, 50 sampling steps - how does performance change?
3. **Data efficiency**: Train with 10, 50, 100, 200 expert episodes - plot learning curves

### Intermediate

4. **Better integration**: Implement RK4 instead of Euler for ODE solving
5. **Stochastic policy**: Add small Gaussian noise at test time for exploration
6. **Conditional generation**: Add a "target angle" and train the model to reach it

### Advanced

7. **Classifier-free guidance**: Learn unconditional and conditional models, then combine
8. **Flow matching + RL**: Use flow matching policy as initialization for PPO
9. **Port to humanoid**: Adapt this code to `dm_control humanoid` environment

## 📚 Key Takeaways

1. **Flow Matching = Learning Vector Fields**: We learn which direction to move in action space

2. **Simple Loss**: Just predict the direction from noise to expert action

3. **Flexible Framework**: Easy to add conditions (velocity targets, goals, etc.)

4. **Scales Well**: Same algorithm works for 1D pendulum and 30D humanoid

5. **Smooth Actions**: Continuous-time formulation ensures smooth control

## 🔗 Further Reading

### Papers
- **Flow Matching for Generative Modeling** (Lipman et al., 2023): Original flow matching paper
- **Diffusion Policy** (Chi et al., 2023): Diffusion for robot learning
- **Recurrent Flow Matching** (Khader et al., 2024): For sequential decisions

### Related Work
- **Diffusion Models**: Related but uses noise schedules instead of straight paths
- **Score Matching**: Another way to learn generative models
- **Normalizing Flows**: Older approach, requires invertible architectures

---

**Congratulations!** 🎉 You now understand flow matching for control. You're ready to tackle complex locomotion tasks!

For questions or contributions, open an issue on GitHub.
