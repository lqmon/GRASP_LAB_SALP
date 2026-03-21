#!/usr/bin/env python3
"""
SALP Unified Training Script

Simple, config-driven training. All parameters come from YAML files.

Usage:
    # Use default config
    python train.py
    
    # Use a preset config
    python train.py --config single_food
    
    # Use custom config file
    python train.py --config path/to/custom.yaml
    
    # With visual feedback
    python train.py --visual
    
    # Use SB3 implementation (default is custom)
    python train.py --sb3
    
    # Continue from checkpoint
    python train.py --checkpoint data/models/best_model.zip
    
    # Override config values
    python train.py --config single_food --timesteps 50000 --eval-freq 1000
"""

import argparse
import sys
import os
from pathlib import Path
import logging
import json
from datetime import datetime
import traceback

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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from salp.config.config_loader import load_config
from salp.environments.salp_snake_env import SalpSnakeEnv


def train_sb3(config, args):
    """Train using Stable Baselines3 SAC."""
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
        import numpy as np
        from datetime import datetime
        
        logger.info("=" * 70)
        logger.info("SALP TRAINING - Stable Baselines3 SAC")
        logger.info("=" * 70)
        
        # Validate configuration
        env_params = config.environment.get('params', {})
        logger.info(f"Environment Config: {config.environment}")
        logger.info(f"Agent Config: {config.agent}")
        logger.info(f"Training Config: {config.training}")
        
        # Create training environment
        logger.info("Creating training environment...")
        try:
            train_env = SalpSnakeEnv(render_mode=None, **env_params)
            logger.info(f"✓ Training environment created: {train_env}")
        except Exception as e:
            logger.error(f"✗ Failed to create training environment: {e}")
            raise
        
        # Create evaluation environment
        logger.info("Creating evaluation environment...")
        try:
            eval_env = SalpSnakeEnv(render_mode=None, **env_params)
            logger.info(f"✓ Evaluation environment created")
        except Exception as e:
            logger.error(f"✗ Failed to create evaluation environment: {e}")
            train_env.close()
            raise
        
        # Test environment step
        logger.info("Testing environment step...")
        try:
            obs, info = train_env.reset()
            action = train_env.action_space.sample()
            obs, reward, terminated, truncated, info = train_env.step(action)
            logger.info(f"✓ Environment step successful")
            logger.info(f"  - Observation shape: {obs.shape}")
            logger.info(f"  - Reward: {reward:.4f}")
            logger.info(f"  - Terminated: {terminated}, Truncated: {truncated}")
        except Exception as e:
            logger.error(f"✗ Environment step failed: {e}")
            train_env.close()
            eval_env.close()
            raise
        
        # Create or load agent
        logger.info("\nInitializing SAC agent...")
        try:
            if args.checkpoint:
                logger.info(f"Loading checkpoint: {args.checkpoint}")
                agent = SAC.load(args.checkpoint, env=train_env)
                logger.info(f"✓ Checkpoint loaded successfully")
            else:
                agent_config = config.agent
                logger.info(f"Creating new SAC agent with config: {agent_config}")
                agent = SAC(
                    policy="MlpPolicy",
                    env=train_env,
                    learning_rate=agent_config.get('learning_rate', 3e-4),
                    buffer_size=agent_config.get('buffer_size', 500000),
                    batch_size=agent_config.get('batch_size', 128),
                    tau=agent_config.get('tau', 0.005),
                    gamma=agent_config.get('gamma', 0.99),
                    policy_kwargs=dict(net_arch=agent_config.get('hidden_sizes', [256, 256])),
                    verbose=0 if args.visual else 1,
                )
                logger.info(f"✓ SAC agent created with policy network")
        except Exception as e:
            logger.error(f"✗ Failed to initialize agent: {e}")
            train_env.close()
            eval_env.close()
            raise
        
        # Setup save directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = config.training.get('experiment_name', 'salp_training')
        save_dir = Path(config.training.get('model_dir', 'data/models')) / f"{exp_name}_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_path = save_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump({
                'environment': dict(config.environment),
                'agent': dict(config.agent),
                'training': dict(config.training),
                'timestamp': timestamp,
                'command_args': vars(args)
            }, f, indent=2)
        logger.info(f"✓ Configuration saved to {config_path}")
        
        logger.info(f"\n📁 Save directory: {save_dir}")
        
        if args.visual:
            _train_sb3_visual(agent, train_env, eval_env, save_dir, config, args)
        else:
            _train_sb3_basic(agent, train_env, save_dir, config, args)
        
        train_env.close()
        eval_env.close()
        
        logger.info(f"\n✓ Training complete!")
        logger.info(f"  Models saved to: {save_dir}")
        logger.info(f"  Log file: training.log")
        
    except Exception as e:
        logger.error(f"\n✗ Training failed with error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


def _train_sb3_visual(agent, train_env, eval_env, save_dir, config, args):
    """Train SB3 with continuous visual feedback."""
    import time
    import numpy as np
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    
    # Import visual trainer
    sys.path.append(str(Path(__file__).parent / 'scripts' / 'utilities'))
    from continuous_visual_trainer import ContinuousVisualTrainer
    
    env_params = config.environment.get('params', {})
    visual_trainer = ContinuousVisualTrainer(
        env_creator=lambda: SalpSnakeEnv(render_mode="human", **env_params),
        fps=30.0,
        verbose=True
    )
    
    def train_with_visual(vt):
        vt.update_model(agent)
        best_reward = -float('inf')
        
        def evaluate(model, n_episodes=5):
            rewards = []
            for _ in range(n_episodes):
                obs, _ = eval_env.reset()
                done = False
                ep_reward = 0
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = eval_env.step(action)
                    done = terminated or truncated
                    ep_reward += reward
                rewards.append(ep_reward)
            return np.mean(rewards)
        
        class VisualCallback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.eval_freq = config.training.get('eval_frequency', 5000)
                if args.eval_freq:
                    self.eval_freq = args.eval_freq
                    
            def _on_step(self):
                nonlocal best_reward
                if self.n_calls % self.eval_freq == 0 and self.n_calls > 0:
                    mean_reward = evaluate(self.model)
                    print(f"\n📊 Step {self.n_calls:,}: Reward {mean_reward:.1f}")
                    
                    vt.update_training_info({'step': f'{self.n_calls:,}', 'best': f'{best_reward:.1f}'})
                    
                    if mean_reward > best_reward:
                        best_reward = mean_reward
                        best_path = save_dir / "best_model"
                        self.model.save(str(best_path))
                        new_model = SAC.load(str(best_path), env=train_env)
                        vt.update_model(new_model)
                        print("🏆 NEW BEST!")
                return True
        
        timesteps = args.timesteps if args.timesteps else config.training.get('max_episodes', 1000) * 1000
        agent.learn(total_timesteps=timesteps, callback=VisualCallback(), log_interval=None)
        agent.save(str(save_dir / "final_model"))
        vt.stop()
    
    visual_trainer.start_training_thread(train_with_visual)
    time.sleep(2)
    
    try:
        visual_trainer.run_visual_loop()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
        visual_trainer.stop()


def _train_sb3_basic(agent, train_env, save_dir, config, args):
    """Basic SB3 training without visualization with improved logging."""
    from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
    import numpy as np
    
    # Create directories for logs
    log_dir = save_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    checkpoint_dir = save_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Setup callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=10000,
        save_path=str(checkpoint_dir),
        name_prefix="checkpoint"
    )
    
    # Custom callback for detailed logging
    class DetailedLoggingCallback(BaseCallback):
        def __init__(self, log_file):
            super().__init__()
            self.log_file = log_file
            self.episode_rewards = []
            self.episode_lengths = []
            
        def _on_step(self):
            # Log every evaluation step
            if 'episode' in self.locals:
                episode_reward = self.locals['episode']['r']
                episode_length = self.locals['episode']['l']
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                if len(self.episode_rewards) % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:])
                    avg_length = np.mean(self.episode_lengths[-10:])
                    logger.info(f"Episode {len(self.episode_rewards):4d} | Reward: {avg_reward:7.2f} | Steps: {avg_length:6.0f}")
                    
                    # Save statistics
                    stats = {
                        'episode': len(self.episode_rewards),
                        'avg_reward': float(avg_reward),
                        'avg_steps': float(avg_length),
                        'timesteps': self.num_timesteps
                    }
                    with open(self.log_file / f"episode_{len(self.episode_rewards)}.json", 'w') as f:
                        json.dump(stats, f)
            
            return True
    
    logging_cb = DetailedLoggingCallback(log_dir)
    
    timesteps = args.timesteps if args.timesteps else config.training.get('max_episodes', 1000) * 1000
    logger.info(f"\n🚀 Starting training for {timesteps:,} timesteps...")
    logger.info(f"   Checkpoints will be saved every 10,000 steps")
    logger.info(f"   Directory: {save_dir}\n")
    
    try:
        agent.learn(
            total_timesteps=timesteps,
            callback=[checkpoint_cb, logging_cb],
            log_interval=10
        )
        agent.save(str(save_dir / "final_model"))
        logger.info(f"✓ Final model saved to {save_dir / 'final_model'}")
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Training interrupted by user")
        agent.save(str(save_dir / "interrupted_model"))
        logger.info(f"Model saved to {save_dir / 'interrupted_model'}")
    except Exception as e:
        logger.error(f"✗ Training failed: {e}")
        logger.error(traceback.format_exc())
        raise


def train_custom(config, args):
    """Train using custom SAC/GAIL implementation."""
    try:
        from salp.training.continuous_trainer import ContinuousTrainer
        from salp.config.base_config import ExperimentConfig, EnvironmentConfig, AgentConfig, TrainingConfig, GAILConfig
        
        logger.info("=" * 70)
        logger.info("SALP TRAINING - Custom Implementation")
        logger.info("=" * 70)
        
        # Convert new config format to old format (temporary bridge)
        logger.info("Converting configuration format...")
        env_cfg = EnvironmentConfig(
            name=config.environment.get('name'),
            type=config.environment.get('type'),
            width=config.environment.get('width', 800),
            height=config.environment.get('height', 600),
            params=config.environment.get('params', {})
        )
        
        agent_cfg = AgentConfig(
            name=config.agent.get('name'),
            type=config.agent.get('type'),
            hidden_sizes=config.agent.get('hidden_sizes', [256, 256]),
            learning_rate=config.agent.get('learning_rate', 3e-4),
            batch_size=config.agent.get('batch_size', 128),
            buffer_size=config.agent.get('buffer_size', 500000),
            gamma=config.agent.get('gamma', 0.99),
            tau=config.agent.get('tau', 0.005),
            params=config.agent.get('params', {})
        )
        
        training_cfg = TrainingConfig(
            max_episodes=config.training.get('max_episodes', 1000),
            max_steps_per_episode=config.training.get('max_steps_per_episode', 5000),
            eval_frequency=config.training.get('eval_frequency', 25),
            save_frequency=config.training.get('save_frequency', 50),
            start_training_after=config.training.get('start_training_after', 500),
            log_dir=config.training.get('log_dir', 'data/logs'),
            model_dir=config.training.get('model_dir', 'data/models'),
            experiment_name=config.training.get('experiment_name', 'salp_training')
        )
        
        gail_cfg = None
        if config.gail and config.gail.get('use_gail', False):
            logger.info("GAIL training enabled")
            gail_cfg = GAILConfig(**config.gail)
        
        logger.info("✓ Configuration converted successfully")
        logger.info(f"  - Environment: {env_cfg.name}")
        logger.info(f"  - Agent: {agent_cfg.name}")
        logger.info(f"  - Max Episodes: {training_cfg.max_episodes}")
        logger.info(f"  - GAIL: {'Enabled' if gail_cfg else 'Disabled'}\n")
        
        exp_config = ExperimentConfig(env_cfg, agent_cfg, training_cfg, gail_cfg)
        
        logger.info("Initializing trainer...")
        trainer = ContinuousTrainer(exp_config)
        logger.info("✓ Trainer initialized, starting training...\n")
        
        trainer.train()
        
        logger.info("\n✓ Custom training complete!")
        
    except ImportError as e:
        logger.error(f"✗ Failed to import training modules: {e}")
        logger.info("  Make sure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        logger.error(f"✗ Custom training failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


def main():
    """Main entry point with validation and error handling."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--config', '-c', default='defaults', help='Config file or preset name')
    parser.add_argument('--sb3', action='store_true', help='Use Stable Baselines3 (default: custom)')
    parser.add_argument('--visual', '-v', action='store_true', help='Enable continuous visual feedback')
    parser.add_argument('--checkpoint', help='Path to checkpoint to continue from')
    parser.add_argument('--timesteps', '-t', type=int, help='Override total timesteps')
    parser.add_argument('--eval-freq', type=int, help='Override evaluation frequency')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set debug logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate checkpoint path
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            logger.error(f"✗ Checkpoint file not found: {args.checkpoint}")
            sys.exit(1)
        logger.info(f"✓ Checkpoint found: {args.checkpoint}")
    
    # Validate timesteps
    if args.timesteps and args.timesteps <= 0:
        logger.error("✗ Timesteps must be positive")
        sys.exit(1)
    
    # Validate eval frequency
    if args.eval_freq and args.eval_freq <= 0:
        logger.error("✗ Eval frequency must be positive")
        sys.exit(1)
    
    try:
        # Load configuration
        logger.info(f"Loading configuration: {args.config}")
        overrides = {}
        if args.timesteps:
            overrides['training'] = {'max_episodes': args.timesteps // 1000}
        if args.eval_freq:
            overrides.setdefault('training', {})['eval_frequency'] = args.eval_freq
        
        config = load_config(args.config, **overrides)
        logger.info(f"✓ Configuration loaded successfully")
        
    except FileNotFoundError as e:
        logger.error(f"✗ Configuration file not found: {e}")
        logger.info("Available configs: defaults, single_food, single_food_long_horizon, sac_gail")
        sys.exit(1)
    except Exception as e:
        logger.error(f"✗ Failed to load configuration: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    # Print configuration summary
    logger.info("\n" + "=" * 70)
    logger.info("SALP TRAINING CONFIGURATION")
    logger.info("=" * 70)
    logger.info(f"Config File:     {args.config}")
    logger.info(f"Implementation:  {'Stable Baselines3' if args.sb3 else 'Custom'}")
    logger.info(f"Visual Feedback: {'Yes' if args.visual else 'No'}")
    logger.info(f"Checkpoint:      {args.checkpoint if args.checkpoint else 'None (starting fresh)'}")
    logger.info("=" * 70 + "\n")
    
    # Train
    try:
        if args.sb3:
            train_sb3(config, args)
        else:
            train_custom(config, args)
    except KeyboardInterrupt:
        logger.warning("\n\n⚠️ Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n✗ Training failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
