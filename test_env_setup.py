#!/usr/bin/env python3
"""Quick test to verify food initialization and basic environment setup"""
import sys
sys.path.insert(0, 'src')
from salp.environments.salp_snake_env import SalpSnakeEnv
import numpy as np

print("Creating environment...")
env = SalpSnakeEnv(
    forced_breathing=True,
    num_food_items=3,
    width=800,
    height=600,
    render_mode=None
)

print(f"✓ Environment created")
print(f"  width={env.width}, height={env.height}")
print(f"  pixel_scale={env.pixel_scale}")
print(f"  tank_margin={env.tank_margin}")
print(f"  pos_init (tank center in pixels)={env.pos_init}")

print(f"\nResetting environment (this may take ~30 seconds due to physics initialization)...")
obs, info = env.reset()
print(f"✓ Environment reset complete")

print(f"\nRobot state at reset:")
print(f"  robot_pos (in meters): {env.robot_pos}")
print(f"  robot_angle: {env.robot_angle:.3f}")
print(f"  ellipse_a: {env.ellipse_a:.4f}")
print(f"  ellipse_b: {env.ellipse_b:.4f}")

# Convert to pixels
robot_pos_pixels = np.array([
    env.pos_init[0] + env.robot_pos[0] * env.pixel_scale,
    env.pos_init[1] + env.robot_pos[1] * env.pixel_scale
])
print(f"  robot_pos (in pixels): {robot_pos_pixels}")

print(f"\nFood state at reset:")
print(f"  num_food_items: {env.num_food_items}")
print(f"  food_positions (in pixels):")
for i, pos in enumerate(env.food_positions):
    if pos is not None:
        dist_to_robot = np.sqrt((pos[0] - robot_pos_pixels[0])**2 + (pos[1] - robot_pos_pixels[1])**2)
        print(f"    [{i}]: {pos} (distance to robot: {dist_to_robot:.1f} pixels)")
    else:
        print(f"    [{i}]: None")

print(f"\nObservation info:")
print(f"  obs shape: {obs.shape}")
print(f"  obs dtype: {obs.dtype}")

print(f"\nTest complete!")
