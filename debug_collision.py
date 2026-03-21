#!/usr/bin/env python3
"""Debug collision detection issue."""

from src.salp.environments.salp_snake_env import SalpSnakeEnv
import numpy as np

env = SalpSnakeEnv(render_mode=None, num_food_items=5, forced_breathing=True)
obs, info = env.reset()

print(f"Initial robot position: {env.robot_pos}")
print(f"Initial robot angle: {env.robot_angle}")
print(f"Initial ellipse_a: {env.ellipse_a}")
print(f"Initial ellipse_b: {env.ellipse_b}")
print(f"Tank margin (pixels): {env.tank_margin}")
print(f"Width x Height: {env.width} x {env.height}")
print(f"Pixel scale: {env.pixel_scale}")

# Convert robot position to pixels
robot_pos_pixels = env.robot_pos * env.pixel_scale
print(f"\nRobot position in pixels: {robot_pos_pixels}")

# Check collision
robot_radius_pixels = max(env.ellipse_a, env.ellipse_b) * env.pixel_scale
print(f"Robot radius in pixels: {robot_radius_pixels}")

# Check each boundary
margin = env.tank_margin
print(f"\nBoundary checks:")
print(f"  Left:   {robot_pos_pixels[0]:.2f} - {robot_radius_pixels:.2f} = {robot_pos_pixels[0] - robot_radius_pixels:.2f} <= {margin}? {robot_pos_pixels[0] - robot_radius_pixels <= margin}")
print(f"  Right:  {robot_pos_pixels[0]:.2f} + {robot_radius_pixels:.2f} = {robot_pos_pixels[0] + robot_radius_pixels:.2f} >= {env.width - margin}? {robot_pos_pixels[0] + robot_radius_pixels >= env.width - margin}")
print(f"  Top:    {robot_pos_pixels[1]:.2f} - {robot_radius_pixels:.2f} = {robot_pos_pixels[1] - robot_radius_pixels:.2f} <= {margin}? {robot_pos_pixels[1] - robot_radius_pixels <= margin}")
print(f"  Bottom: {robot_pos_pixels[1]:.2f} + {robot_radius_pixels:.2f} = {robot_pos_pixels[1] + robot_radius_pixels:.2f} >= {env.height - margin}? {robot_pos_pixels[1] + robot_radius_pixels >= env.height - margin}")

# Take a step
print("\nTaking a step...")
action = np.array([0.5])
obs, r, term, trunc, info = env.step(action)

print(f"\nAfter step:")
print(f"Robot position: {env.robot_pos}")
print(f"Robot position in pixels: {env.robot_pos * env.pixel_scale}")
print(f"Collision: {info['collision']}")
print(f"Reward: {r}")
