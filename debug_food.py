#!/usr/bin/env python3
"""Debug food collection and robot movement."""

from src.salp.environments.salp_snake_env import SalpSnakeEnv
import numpy as np

env = SalpSnakeEnv(render_mode=None, num_food_items=5, forced_breathing=True, max_steps_without_food=500)
obs, info = env.reset()

print("Debug Info:")
print(f"Robot position (meters): {env.robot_pos}")
print(f"Robot position (pixels): {env.pos_init[0] + env.robot_pos[0] * env.pixel_scale}, {env.pos_init[1] + env.robot_pos[1] * env.pixel_scale}")
print(f"Food positions (pixels):")
for i, food in enumerate(env.food_positions):
    if food:
        dist_pixels = np.sqrt((food[0] - env.pos_init[0])**2 + (food[1] - env.pos_init[1])**2)
        print(f"  Food {i}: {food} (distance: {dist_pixels:.1f} pixels)")

print("\nTesting robot movement with non-random actions...")
# Try steering aggressively toward first food
for step in range(50):
    action = np.array([1.0])  # Max nozzle right
    obs, r, term, trunc, info = env.step(action)
    
    if step % 10 == 0:
        robot_pos_px = env.pos_init[0] + env.robot_pos[0] * env.pixel_scale
        robot_pos_py = env.pos_init[1] + env.robot_pos[1] * env.pixel_scale
        print(f"Step {step:2d}: Robot position: ({robot_pos_px:.1f}, {robot_pos_py:.1f}), Reward: {r:.3f}")

print(f"\nAfter 50 steps:")
print(f"  Robot moved: {np.linalg.norm([env.robot_pos[0], env.robot_pos[1]]):.4f} meters")
print(f"  Food collected: {info['food_collected']}")
