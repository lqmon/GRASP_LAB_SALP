#!/usr/bin/env python3
"""
Test script to verify food collection mechanics with actual environment.
Runs a single episode with logging to see what's happening.
"""
import sys
sys.path.insert(0, 'src')
from salp.environments.salp_snake_env import SalpSnakeEnv
import numpy as np

# Create minimal environment
env = SalpSnakeEnv(
    render_mode=None,
    width=400,
    height=300,
    num_food_items=1,
    forced_breathing=True,
    food_reward=10.0,
    collision_penalty=-5.0,
    time_penalty=-0.01,
)

print("Environment created. Resetting...")
obs, info = env.reset()

print(f"\n=== INITIAL STATE ===")
print(f"Robot position (meters): {env.robot_pos}")
print(f"Food positions (pixels): {env.food_positions}")
print(f"Observation shape: {obs.shape}")
print(f"Observation dtype: {obs.dtype}")

# Run a few steps
print(f"\n=== RUNNING 20 STEPS ===")
for step in range(20):
    # Take a random action
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    
    if step % 5 == 0:
        robot_pos_pixels = np.array([
            env.pos_init[0] + env.robot_pos[0] * env.pixel_scale,
            env.pos_init[1] + env.robot_pos[1] * env.pixel_scale
        ])
        print(f"Step {step:2d}: robot_pixels={robot_pos_pixels.astype(int)}, "
              f"reward={reward:.4f}, done={done}, truncated={truncated}, "
              f"food_collected={info.get('food_collected', 0)}, "
              f"collision={info.get('collision', False)}")
    
    if done or truncated:
        print(f"  Episode ended at step {step}")
        break

print(f"\n✓ Test complete!")
