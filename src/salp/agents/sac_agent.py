"""
SAC Agent for SALP training.
Wrapper around Stable Baselines3 SAC implementation.
"""

import numpy as np
from typing import Dict, Any, Optional
import torch
from stable_baselines3 import SAC
from salp.config.base_config import AgentConfig


class SACAgent:
    """
    Soft Actor-Critic (SAC) agent using Stable Baselines3.
    """
    
    def __init__(self, config: AgentConfig, obs_dim: int, action_dim: int, action_space, verbose: int = 0):
        """
        Initialize SAC agent.
        
        Args:
            config: Agent configuration
            obs_dim: Observation dimension
            action_dim: Action dimension
            action_space: Gymnasium action space
            verbose: Verbosity level
        """
        self.config = config
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_space = action_space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create dummy environment for SAC initialization
        # SAC needs an env object but we'll set learn via train_agent method
        self.dummy_env = None
        self.model = None
        self._initialized = False
    
    def _initialize_model(self, env):
        """Initialize the SAC model with an environment."""
        if self._initialized:
            return
        
        # Create SB3 SAC agent
        self.model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=self.config.learning_rate,
            buffer_size=self.config.buffer_size,
            learning_starts=self.config.params.get('learning_starts', 1000),
            batch_size=self.config.batch_size,
            tau=self.config.tau,
            gamma=self.config.gamma,
            train_freq=self.config.params.get('train_freq', 1),
            gradient_steps=self.config.params.get('gradient_steps', 1),
            ent_coef=self.config.params.get('alpha', 'auto'),
            target_update_interval=self.config.params.get('target_update_interval', 1),
            target_entropy=self.config.params.get('target_entropy', 'auto'),
            use_sde=self.config.params.get('use_sde', False),
            policy_kwargs=dict(net_arch=self.config.hidden_sizes),
            verbose=0,
            device=self.device
        )
        self._initialized = True
    
    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action from observation."""
        if self.model is None:
            # Return random action if not initialized
            return self.action_space.sample()
        return self.model.predict(observation, deterministic=deterministic)[0]
    
    def update(self, batch: Dict) -> Dict[str, float]:
        """Update agent with a batch of transitions."""
        if self.model is None:
            return {}
        
        # Convert batch to proper format for SB3
        obs = batch['observations'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_obs = batch['next_observations'].to(self.device)
        dones = batch['dones'].to(self.device)
        
        # SB3 SAC handles its own training, so we return empty dict
        # In practice, you'd use the model's train_frequency and num_steps
        return {}
    
    def train(self, env, total_timesteps: int, callback=None):
        """Train the agent."""
        if self.model is None:
            self._initialize_model(env)
        
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
    
    def save(self, path: str):
        """Save model to disk."""
        if self.model is not None:
            self.model.save(path)
    
    def load(self, path: str):
        """Load model from disk."""
        self.model = SAC.load(path)
        self._initialized = True
