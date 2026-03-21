"""
Interactive keyboard control for the SALP robot environment.

This script allows you to manually control the robot's motion using keyboard input.
You can adjust the inhale control, nozzle steering, and coast time in real-time.

Run this script and follow the on-screen instructions to control the robot.
"""

import sys
import os
import numpy as np
import pygame

# Add project root to path
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from salp_robot_env import SalpRobotEnv
from salp.environments.robot import Robot, Nozzle


def main():
    """Run the interactive robot control demo."""
    
    # Initialize Pygame
    pygame.init()
    
    # Create robot with default parameters
    nozzle = Nozzle(length1=0.05, length2=0.05, length3=0.05, area=0.00016, mass=1.0)
    robot = Robot(
        dry_mass=1.0, 
        init_length=0.3, 
        init_width=0.15, 
        max_contraction=0.06, 
        nozzle=nozzle
    )
    robot.nozzle.set_angles(angle1=0.0, angle2=np.pi)
    robot.set_environment(density=1000)  # water density in kg/m^3
    
    # Create environment with rendering enabled
    env = SalpRobotEnv(render_mode="human", width=900, height=700, robot=robot)
    obs, info = env.reset()
    
    print("\n" + "="*70)
    print(" SALP ROBOT INTERACTIVE KEYBOARD CONTROL")
    print("="*70)
    print("\nStarting interactive control session...\n")
    
    try:
        # Run interactive control
        env.interactive_control(max_cycles=None)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        env.close()
        pygame.quit()
        print("Cleaned up resources")


if __name__ == "__main__":
    main()
