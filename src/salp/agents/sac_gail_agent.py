"""
SAC + GAIL Agent for SALP training with imitation learning.
"""

import numpy as np
from typing import Dict, Any, Optional
import torch
from stable_baselines3 import SAC
from salp.config.base_config import AgentConfig
from salp.agents.discriminator import Discriminator


class SACGAILAgent:
    """
    SAC agent with GAIL (Generative Adversarial Imitation Learning) component.
    """
    
    def __init__(self, config: AgentConfig, obs_dim: int, action_dim: int, action_space, 
                 expert_buffer=None, verbose: int = 0):
        """
        Initialize SAC + GAIL agent.
        
        Args:
            config: Agent configuration
            obs_dim: Observation dimension
            action_dim: Action dimension
            action_space: Gymnasium action space
            expert_buffer: Expert demonstration buffer
            verbose: Verbosity level
        """
        self.config = config
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_space = action_space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.expert_buffer = expert_buffer
        
        # SAC base agent
        self.model = None
        self._initialized = False
        
        # GAIL discriminator
        self.discriminator = None
        self._setup_discriminator()
    
    def _setup_discriminator(self):
        """Setup the GAIL discriminator."""
        if self.expert_buffer is None or len(self.expert_buffer) == 0:
            return
        
        # Create discriminator network
        hidden_sizes = [256, 256]
        self.discriminator = Discriminator(
            input_dim=self.obs_dim + self.action_dim,
            output_dim=1,
            hidden_sizes=hidden_sizes
        )
        self.discriminator.to(self.device)
    
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
            return self.action_space.sample()
        return self.model.predict(observation, deterministic=deterministic)[0]
    
    def update(self, batch: Dict) -> Dict[str, float]:
        """Update agent with a batch of transitions."""
        if self.model is None or self.discriminator is None:
            return {}
        
        # Convert batch to tensors
        obs = batch['observations'].to(self.device)
        actions = batch['actions'].to(self.device)
        
        # GAIL discriminator training
        # Compute rewards using discriminator
        with torch.no_grad():
            state_action = torch.cat([obs, actions], dim=-1)
            discriminator_output = self.discriminator(state_action)
            # Reward is log probability of agent action
            gail_reward = torch.log(discriminator_output + 1e-8)
        
        return {'gail_reward': gail_reward.mean().item()}
    
    def train(self, env, total_timesteps: int, callback=None):
        """Train the agent."""
        if self.model is None:
            self._initialize_model(env)
        
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
    
    def save(self, path: str):
        """Save model to disk."""
        if self.model is not None:
            self.model.save(path)
        if self.discriminator is not None:
            discriminator_path = path.replace('.zip', '_discriminator.pt')
            torch.save(self.discriminator.state_dict(), discriminator_path)
    
    def load(self, path: str):
        """Load model from disk."""
        self.model = SAC.load(path)
        self._initialized = True
        
        if self.discriminator is not None:
            discriminator_path = path.replace('.zip', '_discriminator.pt')
            try:
                self.discriminator.load_state_dict(torch.load(discriminator_path))
            except FileNotFoundError:
                print(f"Warning: Could not find discriminator checkpoint at {discriminator_path}")
