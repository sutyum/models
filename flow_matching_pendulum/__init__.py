"""
Flow Matching for Inverted Pendulum Control

A complete educational implementation of flow matching applied to
reinforcement learning control problems.
"""

from .flow_matching import FlowMatching, VelocityNet
from .pendulum_env import PendulumEnv, ExpertController, collect_expert_trajectories
from .train import FlowMatchingPolicy, train_flow_matching_policy, evaluate_policy

__all__ = [
    "FlowMatching",
    "VelocityNet",
    "PendulumEnv",
    "ExpertController",
    "collect_expert_trajectories",
    "FlowMatchingPolicy",
    "train_flow_matching_policy",
    "evaluate_policy",
]
