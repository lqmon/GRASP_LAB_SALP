# test_robot.py
from stable_baselines3 import SAC
from salp_robot_env import SalpRobotEnv
from salp.environments.robot import Robot, Nozzle

nozzle = Nozzle(length1=0.05, length2=0.05, length3=0.05, area=0.00016, mass=1.0)
robot = Robot(dry_mass=1.0, init_length=0.3, init_width=0.15, 
                  max_contraction=0.06, nozzle=nozzle)
robot.nozzle.set_angles(angle1=0.0, angle2=0.0)
env = SalpRobotEnv(render_mode="human", robot=robot)
# Load the trained model
model = SAC.load("./logs/salp_robot_model_400000_steps", env=env)   


obs, _ = env.reset()
env.start_recording()
for i in range(100):
    # predict() returns the action and the hidden state (unused for MlpPolicy)
    action, _states = model.predict(obs, deterministic=True)
    
    obs, reward, terminated, truncated, info = env.step(action)

    env.wait_for_animation()
    
    # Print or render here
    print(f"Step {i}: Action={action}, State={obs}")
    
    if truncated:
        obs, _ = env.reset()
    
    if terminated:
        print("Episode finished!")
        break
gif_path = env.stop_recording("test_robot_simulation2.gif")
env.close()
print(f"Simulation GIF saved to: {gif_path}")