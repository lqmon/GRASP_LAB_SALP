#!/usr/bin/env python3
"""Test fast_test config to verify food can be collected."""

from src.salp.environments.salp_snake_env import SalpSnakeEnv
import numpy as np

print("Testing fast_test config (smaller tank, fewer food items)...\n")

env = SalpSnakeEnv(
    render_mode=None, 
    width=400,  # Smaller tank
    height=400,
    num_food_items=3,
    forced_breathing=True,
    max_steps_without_food=2000
)

obs, info = env.reset()

print(f"Tank size: 400x400 (smaller)")
print(f"Robot starts at: (200, 200)")
print(f"Food spawned at:")
for i, food in enumerate(env.food_positions):
    if food:
        dist = np.sqrt((food[0] - 200)**2 + (food[1] - 200)**2)
        print(f"  Food {i}: ({food[0]:.0f}, {food[1]:.0f}) - distance: {dist:.0f} pixels")

print(f"\n Trying random actions for 2000 steps...")
episode_steps = 0
total_reward = 0
max_food = 0

for i in range(2000):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    episode_steps += 1
    total_reward += r
    max_food = max(max_food, info['food_collected'])
    
    if term or trunc:
        break

print(f"\n✓ Episode ran for {episode_steps} steps")
print(f"  Total reward: {total_reward:.2f}")
print(f"  Food collected: {info['food_collected']}")
print(f"  Max food in episode: {max_food}")

if max_food > 0:
    print("\n✅ GOOD! Food CAN be collected with this config")
else:
    print("\n⚠️  Still can't collect food - need even longer episodes or more aggressive actions")
