#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')
from salp.environments.salp_snake_env import SalpSnakeEnv
import numpy as np

env = SalpSnakeEnv(forced_breathing=True, num_food_items=1)
obs, info = env.reset()

print(f"Environment initialized")
print(f"  Tank size: {env.width} x {env.height}")
print(f"  pos_init (tank center): {env.pos_init}")
print(f"  tank_margin: {env.tank_margin}")
print(f"Robot initial state:")
print(f"  robot_pos (meters): {env.robot_pos}")
print(f"  robot_angle: {env.robot_angle:.3f}")
print(f"  ellipse_a (length): {env.ellipse_a:.4f}")
print(f"  ellipse_b (width): {env.ellipse_b:.4f}")
print(f"  robot radius (pixels): {max(env.ellipse_a, env.ellipse_b) * env.pixel_scale:.1f}")

print(f"\nFood positions (pixels): {env.food_positions}")

# Manually check collision logic
robot_pos_pixels = np.array([
    env.pos_init[0] + env.robot_pos[0] * env.pixel_scale,
    env.pos_init[1] + env.robot_pos[1] * env.pixel_scale
])
robot_radius = max(env.ellipse_a, env.ellipse_b) * env.pixel_scale
margin = env.tank_margin

print(f"\nCollision detection parameters:")
print(f"  Robot pos in pixels: {robot_pos_pixels}")
print(f"  Robot radius: {robot_radius:.1f}")
print(f"  Expected safe bounds: x in [{margin + robot_radius:.1f}, {env.width - margin - robot_radius:.1f}]")
print(f"                        y in [{margin + robot_radius:.1f}, {env.height - margin - robot_radius:.1f}]")

collision = env._check_wall_collision()
print(f"\n  Initial collision detected: {collision}")

# Try a few steps without stopping
print(f"\nRunning 10 steps:")
for i in range(10):
    action = np.array([0.0])  # Neutral action
    obs, reward, done, truncated, info = env.step(action)
    robot_pos_pixels = np.array([
        env.pos_init[0] + env.robot_pos[0] * env.pixel_scale,
        env.pos_init[1] + env.robot_pos[1] * env.pixel_scale
    ])
    collision_status = info.get('collision', False)
    print(f"  Step {i+1}: reward={reward:.4f}, collision={collision_status}, done={done}, truncated={truncated}")
    if done or truncated:
        print(f"    Episode ended early at step {i+1}!")
        break
