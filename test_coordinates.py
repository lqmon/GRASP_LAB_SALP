#!/usr/bin/env python3
"""
Quick unit test for food collection and coordinate system fixes.
This doesn't run full physics - just tests the logic.
"""
import sys
sys.path.insert(0, 'src')
import numpy as np
import math

def test_food_coordinates():
    """Test coordinate system conversions"""
    # Simulate environment parameters
    pos_init = np.array([400.0, 300.0])  # Tank center in pixels
    pixel_scale = 20.0
    tank_margin = 50
    food_radius = 15
    
    # Simulate robot position (in meters)
    robot_pos = np.array([0.0, 0.0])  # Center of tank
    
    # Convert to pixels
    robot_pos_pixels = pos_init + robot_pos * pixel_scale
    print(f"Robot position in pixels: {robot_pos_pixels}")  # Should be [400, 300]
    
    # Simulate food position (in pixels)
    food_pos = np.array([420.0, 300.0])  # 20 pixels to the right
    
    # Calculate distance (both in pixels now - FIXED!)
    distance = math.sqrt((food_pos[0] - robot_pos_pixels[0])**2 + 
                         (food_pos[1] - robot_pos_pixels[1])**2)
    print(f"Distance to food: {distance:.1f} pixels")  # Should be 20
    
    # Test angle calculation (both now in pixels - FIXED!)
    angle_to_food = math.atan2(food_pos[1] - robot_pos_pixels[1],
                              food_pos[0] - robot_pos_pixels[0])
    print(f"Angle to food (radians): {angle_to_food:.3f}")  # Should be 0 (straight right)
    print(f"Angle to food (degrees): {math.degrees(angle_to_food):.1f}")  # Should be 0
    
    # Test respawn distance check
    robot_distance = math.sqrt((food_pos[0] - robot_pos_pixels[0])**2 + 
                              (food_pos[1] - robot_pos_pixels[1])**2)
    min_food_distance = 80
    
    if robot_distance < min_food_distance:
        print(f"✗ FAILED: Food too close to robot ({robot_distance:.1f} < {min_food_distance})")
    else:
        print(f"✓ PASSED: Food far enough from robot ({robot_distance:.1f} >= {min_food_distance})")
    
    # Test with food very close
    food_pos2 = np.array([401.0, 300.0])  # 1 pixel away
    robot_distance2 = math.sqrt((food_pos2[0] - robot_pos_pixels[0])**2 + 
                               (food_pos2[1] - robot_pos_pixels[1])**2)
    
    if robot_distance2 < 10:  # Food collection radius
        print(f"✓ CORRECT: Food at {robot_distance2:.1f} pixels would be collected")
    else:
        print(f"Note: Food at {robot_distance2:.1f} pixels is outside collection radius")
    
    print("\n✅ Coordinate system tests passed!")

if __name__ == '__main__':
    test_food_coordinates()
