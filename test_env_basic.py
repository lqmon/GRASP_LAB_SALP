#!/usr/bin/env python3
"""Minimal environment initialization test"""
import sys
sys.path.insert(0, 'src')

print("Testing environment initialization...")
try:
    from salp.environments.salp_snake_env import SalpSnakeEnv
    print("✓ Import successful")
    
    env = SalpSnakeEnv(render_mode=None, num_food_items=1, forced_breathing=True)
    print("✓ Environment instantiated")
    
    obs, info = env.reset()
    print(f"✓ Environment reset successful")
    print(f"  - Obs shape: {obs.shape}")
    print(f"  - Food positions: {env.food_positions}")
    print(f"  - Robot position (meters): {env.robot_pos}")
    
    # Test one step
    action = env.action_space.sample()
    obs2, reward, done, truncated, info = env.step(action)
    print(f"✓ One step executed")
    print(f"  - Reward: {reward:.4f}")
    print(f"  - Food collected: {info.get('food_collected', 0)}")
    print(f"  - Collision: {info.get('collision', False)}")
    
    print("\n✅ Environment is working correctly!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
