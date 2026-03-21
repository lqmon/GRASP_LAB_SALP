"""
Base classes for RL agents in the SALP project.
Only includes utilities needed by custom components (like Discriminator).
For standard RL agents, use Stable Baselines3 implementations.
"""

from typing import Optional, List
import torch
import torch.nn as nn


class BaseNetwork(nn.Module):
    """
    Base neural network class with common functionality.
    Used by custom components like Discriminator.
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: List[int], 
                 activation: str = "relu", output_activation: Optional[str] = None):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_sizes = hidden_sizes
        
        # Activation functions
        activations = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "leaky_relu": nn.LeakyReLU,
            "elu": nn.ELU
        }
        
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.activation = activations[activation]()
        self.output_activation = activations[output_activation]() if output_activation else None
        
        # Build network layers
        self.layers = self._build_layers()
    
    def _build_layers(self) -> nn.ModuleList:
        """Build network layers."""
        layers = nn.ModuleList()
        
        # Input layer
        prev_size = self.input_dim
        
        # Hidden layers
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, self.output_dim))
        
        return layers
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Hidden layers with activation
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        
        # Output layer
        x = self.layers[-1](x)
        
        # Output activation if specified
        if self.output_activation is not None:
            x = self.output_activation(x)
        
        return x


def soft_update(target_net: nn.Module, source_net: nn.Module, tau: float):
    """Soft update of target network parameters."""
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


def hard_update(target_net: nn.Module, source_net: nn.Module):
    """Hard update of target network parameters."""
    target_net.load_state_dict(source_net.state_dict())


class BaseAgent:
    """Base class for RL agents."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def select_action(self, observation, deterministic=False):
        """Select an action given an observation."""
        raise NotImplementedError
    
    def update(self, batch):
        """Update agent with a batch of experiences."""
        raise NotImplementedError


class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions."""
    
    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            obs_dim: Observation dimension
            action_dim: Action dimension
        """
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.size = 0
        self.index = 0
        
        # Pre-allocate arrays
        self.observations = torch.zeros(capacity, obs_dim)
        self.actions = torch.zeros(capacity, action_dim)
        self.rewards = torch.zeros(capacity, 1)
        self.next_observations = torch.zeros(capacity, obs_dim)
        self.dones = torch.zeros(capacity, 1)
    
    def add(self, obs, action, reward, next_obs, done):
        """Add a transition to the buffer."""
        # Convert to tensors if needed
        obs = torch.tensor(obs, dtype=torch.float32) if not isinstance(obs, torch.Tensor) else obs
        action = torch.tensor(action, dtype=torch.float32) if not isinstance(action, torch.Tensor) else action
        reward = torch.tensor([reward], dtype=torch.float32) if not isinstance(reward, torch.Tensor) else reward
        next_obs = torch.tensor(next_obs, dtype=torch.float32) if not isinstance(next_obs, torch.Tensor) else next_obs
        done = torch.tensor([done], dtype=torch.float32) if not isinstance(done, torch.Tensor) else done
        
        # Store transition
        self.observations[self.index] = obs
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_observations[self.index] = next_obs
        self.dones[self.index] = done
        
        # Update indices
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int):
        """Sample a random batch of transitions."""
        indices = torch.randperm(self.size)[:batch_size]
        
        batch = {
            'observations': self.observations[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_observations': self.next_observations[indices],
            'dones': self.dones[indices]
        }
        
        return batch
    
    def __len__(self):
        """Return the number of transitions stored."""
        return self.size


class Logger:
    """Logger for tracking training metrics."""
    
    def __init__(self, log_dir: str):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = log_dir
        self.metrics = {}
        self.episodes = []
        self.scalars = {}
        
        # Create log directory
        import os
        os.makedirs(log_dir, exist_ok=True)
    
    def log_episode(self, episode: int, metrics: dict):
        """Log metrics for an episode."""
        self.episodes.append((episode, metrics))
        
        # Print episode info
        print(f"Episode {episode}: ", end="")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"{key}={value:.2f} ", end="")
        print()
    
    def log_scalar(self, name: str, value: float, step: int):
        """Log a scalar metric."""
        if name not in self.scalars:
            self.scalars[name] = []
        self.scalars[name].append((step, value))
    
    def save_metrics(self):
        """Save all logged metrics to file."""
        import json
        import os
        
        # Save episodes
        if self.episodes:
            episodes_file = os.path.join(self.log_dir, "episodes.json")
            with open(episodes_file, 'w') as f:
                # Convert to serializable format
                episodes_data = []
                for episode, metrics in self.episodes:
                    serializable_metrics = {}
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            serializable_metrics[key] = value
                        else:
                            serializable_metrics[key] = str(value)
                    episodes_data.append({"episode": episode, "metrics": serializable_metrics})
                json.dump(episodes_data, f, indent=2)
        
        # Save scalars
        if self.scalars:
            scalars_file = os.path.join(self.log_dir, "scalars.json")
            with open(scalars_file, 'w') as f:
                # Convert tensor steps to regular numbers
                scalars_data = {}
                for name, values in self.scalars.items():
                    scalars_data[name] = [[int(step), float(val)] for step, val in values]
                json.dump(scalars_data, f, indent=2)
