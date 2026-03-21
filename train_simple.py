#!/usr/bin/env python3
"""
Simple, non-visual training script for SALP.
Avoids the hanging visual rendering loop.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from salp.config.config_loader import load_config
from salp.environments.salp_snake_env import SalpSnakeEnv
from stable_baselines3 import SAC
import numpy as np

def train_simple():
    """Simple non-visual training loop"""
    
    logger.info("=" * 70)
    logger.info("SALP SIMPLE NON-VISUAL TRAINING")
    logger.info("=" * 70)
    
    # Load config
    config = load_config('defaults')
    logger.info(f"\n✓ Configuration loaded")
    
    # Create environment
    logger.info("\nCreating environment...")
    try:
        env = SalpSnakeEnv(
            render_mode=None,
            width=config.environment.get('width', 800),
            height=config.environment.get('height', 600),
            **config.environment.get('params', {})
        )
        logger.info(f"✓ Environment created")
    except Exception as e:
        logger.error(f"✗ Environment creation failed: {e}")
        return
    
    # Create agent
    logger.info("\nCreating SAC agent...")
    try:
        agent_cfg = config.agent
        agent = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=agent_cfg.get('learning_rate', 3e-4),
            buffer_size=agent_cfg.get('buffer_size', 100000),
            batch_size=agent_cfg.get('batch_size', 128),
            tau=agent_cfg.get('tau', 0.005),
            gamma=agent_cfg.get('gamma', 0.99),
            policy_kwargs=dict(net_arch=agent_cfg.get('hidden_sizes', [256, 256])),
            verbose=1,
        )
        logger.info(f"✓ SAC agent created")
    except Exception as e:
        logger.error(f"✗ Agent creation failed: {e}")
        env.close()
        return
    
    # Start training
    logger.info("\n" + "=" * 70)
    logger.info("STARTING TRAINING")
    logger.info("=" * 70 + "\n")
    
    max_episodes = config.training.get('max_episodes', 100)
    max_steps_per_episode = config.training.get('max_steps_per_episode', 5000)
    
    try:
        # Total timesteps to train for
        total_timesteps = max_episodes * max_steps_per_episode
        
        # Train
        agent.learn(total_timesteps=total_timesteps, log_interval=1)
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)
        
        # Save model
        model_path = "data/models/salp_trained"
        os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
        agent.save(model_path)
        logger.info(f"✓ Model saved to {model_path}.zip")
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Training interrupted by user")
    except Exception as e:
        logger.error(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        logger.info("\n✓ Environment closed")

if __name__ == "__main__":
    train_simple()
