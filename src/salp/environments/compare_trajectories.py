"""
Compare robot trajectories with different action inputs.

This script simulates the robot with various combinations of:
- Contraction levels
- Coast times
- Nozzle yaw angles

and visualizes the resulting trajectories for comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow

from salp.environments.robot import Robot, Nozzle


def compare_actions_with_states(actions, expected_states, robot=None, verbose=True):
    """Compare simulated trajectory from actions with expected states.
    
    Args:
        actions: Array of shape (n_steps, 3) containing [contraction, coast_time, yaw_angle]
        expected_states: Array of shape (n_steps, 6) containing expected [pos_x, pos_y, vel_x, vel_y, yaw, angular_vel]
        robot: Robot instance to use (if None, creates default robot)
        verbose: Whether to print detailed comparison results
        
    Returns:
        Dictionary containing:
            - actual_states: Simulated states from actions
            - expected_states: Input expected states
            - errors: State-by-state errors
            - position_error: Mean position error (m)
            - velocity_error: Mean velocity error (m/s)
            - angle_error: Mean angular error (rad)
            - max_position_error: Maximum position error (m)
    """
    if robot is None:
        nozzle = Nozzle(length1=0.05, length2=0.05, length3=0.05, area=0.00016, mass=1.0)
        robot = Robot(dry_mass=1.0, init_length=0.3, init_width=0.15, 
                      max_contraction=0.06, nozzle=nozzle)
        robot.set_environment(density=1000)
        robot.nozzle.set_angles(angle1=0.0, angle2=0.0)
    
    robot.reset()
    
    n_steps = len(actions)
    actual_states = []
    
    # Simulate robot with given actions
    for i, action in enumerate(actions):
        contraction, coast_time, yaw_angle = action
        
        robot.nozzle.set_yaw_angle(yaw_angle=yaw_angle)
        robot.nozzle.solve_angles()
        robot.set_control(
            contraction=contraction,
            coast_time=coast_time,
            nozzle_angles=np.array([robot.nozzle.angle1, robot.nozzle.angle2])
        )
        robot.step_through_cycle()
        
        # Get final state after this cycle
        state = np.array([
            robot.position[0],
            robot.position[1],
            robot.velocity[0],
            robot.velocity[1],
            robot.euler_angle[2],
            robot.angular_velocity[2]
        ])
        actual_states.append(state)
    
    actual_states = np.array(actual_states)
    
    # Calculate errors
    errors = actual_states - expected_states
    position_errors = np.linalg.norm(errors[:, 0:2], axis=1)
    velocity_errors = np.linalg.norm(errors[:, 2:4], axis=1)
    angle_errors = np.abs(errors[:, 4])
    angular_vel_errors = np.abs(errors[:, 5])
    
    mean_pos_error = np.mean(position_errors)
    mean_vel_error = np.mean(velocity_errors)
    mean_angle_error = np.mean(angle_errors)
    max_pos_error = np.max(position_errors)
    
    if verbose:
        print("\n" + "=" * 70)
        print("TRAJECTORY COMPARISON: ACTIONS vs EXPECTED STATES")
        print("=" * 70)
        print(f"Number of steps: {n_steps}")
        print(f"\nSummary Statistics:")
        print(f"  Mean position error:     {mean_pos_error:.6f} m")
        print(f"  Mean velocity error:     {mean_vel_error:.6f} m/s")
        print(f"  Mean angle error:        {np.degrees(mean_angle_error):.6f}°")
        print(f"  Max position error:      {max_pos_error:.6f} m")
        print(f"\nStep-by-step comparison:")
        print("-" * 70)
        print(f"{'Step':<6} {'Pos Error (m)':<15} {'Vel Error (m/s)':<18} {'Angle Error (°)':<15}")
        print("-" * 70)
        for i in range(n_steps):
            print(f"{i:<6} {position_errors[i]:<15.6f} {velocity_errors[i]:<18.6f} {np.degrees(angle_errors[i]):<15.6f}")
        print("=" * 70)
    
    return {
        'actual_states': actual_states,
        'expected_states': expected_states,
        'errors': errors,
        'position_errors': position_errors,
        'velocity_errors': velocity_errors,
        'angle_errors': angle_errors,
        'position_error': mean_pos_error,
        'velocity_error': mean_vel_error,
        'angle_error': mean_angle_error,
        'max_position_error': max_pos_error
    }


def simulate_trajectory(robot, n_cycles, contraction, coast_time, yaw_angle):
    """Simulate robot trajectory with given action inputs.
    
    Args:
        robot: Robot instance
        n_cycles: Number of breathing cycles to simulate
        contraction: Contraction distance (m)
        coast_time: Coast phase duration (s)
        yaw_angle: Nozzle yaw angle (radians)
        
    Returns:
        Dictionary containing trajectory data
    """
    robot.reset()
    
    positions = []
    velocities = []
    euler_angles = []
    states = []
    times = []
    
    for i in range(n_cycles):
        robot.nozzle.set_yaw_angle(yaw_angle=yaw_angle)
        robot.nozzle.solve_angles()
        robot.set_control(
            contraction=contraction, 
            coast_time=coast_time, 
            nozzle_angles=np.array([robot.nozzle.angle1, robot.nozzle.angle2])
        )
        robot.step_through_cycle()
        
        # Create time array for this cycle
        cycle_start_time = robot.time - robot.cycle_time
        time_array = np.arange(cycle_start_time, robot.time + robot.dt, robot.dt)[:len(robot.position_history)]
        
        # Accumulate data
        times.extend(time_array)
        positions.extend(robot.position_history)
        velocities.extend(robot.velocity_history)
        euler_angles.extend(robot.euler_angle_history)
        states.extend(robot.state_history)
    
    return {
        'times': np.array(times),
        'positions': np.array(positions),
        'velocities': np.array(velocities),
        'euler_angles': np.array(euler_angles),
        'states': np.array(states)
    }


def plot_trajectory_comparison(trajectories, labels, title="Trajectory Comparison"):
    """Plot multiple trajectories for comparison.
    
    Args:
        trajectories: List of trajectory dictionaries
        labels: List of labels for each trajectory
        title: Plot title
    """
    fig, axes = plt.subplots(1, 1, figsize=(8, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
    
    # Plot 1: X-Y trajectory
    ax = plt.subplot(1, 1, 1)
    for i, (traj, label) in enumerate(zip(trajectories, labels)):
        positions = traj['positions']
        ax.plot(positions[:, 0], positions[:, 1], '-', color=colors[i], 
                label=label, linewidth=2, alpha=0.7)
        # Mark start and end
        ax.plot(positions[0, 0], positions[0, 1], 'o', color=colors[i], 
                markersize=10, markeredgecolor='black')
        ax.plot(positions[-1, 0], positions[-1, 1], 's', color=colors[i], 
                markersize=10, markeredgecolor='black')
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title('X-Y Trajectory', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.axis('equal')
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def compare_contraction_levels():
    """Compare trajectories with different contraction levels."""
    print("Comparing different contraction levels...")
    
    nozzle = Nozzle(length1=0.01, length2=0.01, length3=0.01, area=0.00016, mass=1.0)
    robot = Robot(dry_mass=1.0, init_length=0.3, init_width=0.15, 
                  max_contraction=0.06, nozzle=nozzle)
    robot.set_environment(density=1000)
    robot.nozzle.set_angles(angle1=0.0, angle2=0.0)
    
    contractions = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]  # Different contraction levels
    # contractions = [0.01]  # Different contraction levels
    n_cycles = 1
    coast_time = 1.0
    yaw_angle = 0.0
    
    trajectories = []
    labels = []
    
    for contraction in contractions:
        traj = simulate_trajectory(robot, n_cycles, contraction, coast_time, yaw_angle)
        trajectories.append(traj)
        labels.append(f'Contraction = {contraction:.2f} m')
        final_dist = np.linalg.norm(traj['positions'][-1])
        print(f"  Contraction {contraction:.2f}m: Final distance = {final_dist:.3f} m")
    
    plot_trajectory_comparison(trajectories, labels, "Comparison: Different Contraction Levels")

def compare_coast_times():
    """Compare trajectories with different coast times."""
    print("\nComparing different coast times...")
    
    nozzle = Nozzle(length1=0.01, length2=0.01, length3=0.01, area=0.00016, mass=1.0)
    robot = Robot(dry_mass=1.0, init_length=0.3, init_width=0.15, 
                  max_contraction=0.06, nozzle=nozzle)
    robot.set_environment(density=1000)
    robot.nozzle.set_angles(angle1=0.0, angle2=0.0)
    
    coast_times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]  # Different coast times
    n_cycles = 1
    contraction = 0.06
    yaw_angle = 0.0
    
    trajectories = []
    labels = []
    
    for coast_time in coast_times:
        traj = simulate_trajectory(robot, n_cycles, contraction, coast_time, yaw_angle)
        trajectories.append(traj)
        labels.append(f'Coast Time = {coast_time:.1f} s')
        final_dist = np.linalg.norm(traj['positions'][-1])
        print(f"  Coast time {coast_time:.1f}s: Final distance = {final_dist:.3f} m")

    plot_trajectory_comparison(trajectories, labels, "Comparison: Different Coast Times")

def compare_yaw_angles():
    """Compare trajectories with different yaw angles."""
    print("\nComparing different yaw angles...")
    
    nozzle = Nozzle(length1=0.01, length2=0.01, length3=0.01, area=0.00016, mass=1.0)
    robot = Robot(dry_mass=1.0, init_length=0.3, init_width=0.15, 
                  max_contraction=0.06, nozzle=nozzle)
    robot.set_environment(density=1000)
    robot.nozzle.set_angles(angle1=0.0, angle2=0.0)
    
    yaw_angles = [-np.pi/2, -np.pi/4, -np.pi/8, np.pi/16, np.pi/32, 0.0, np.pi/32, np.pi/16, np.pi/8, np.pi/4, np.pi/2]  # Different yaw angles
    n_cycles = 1
    contraction = 0.06
    coast_time = 10.0
    
    trajectories = []
    labels = []
    
    for yaw_angle in yaw_angles:
        traj = simulate_trajectory(robot, n_cycles, contraction, coast_time, yaw_angle)
        trajectories.append(traj)
        labels.append(f'Yaw = {np.degrees(yaw_angle):.0f}°')  
        final_pos = traj['positions'][-1]
        final_dist = np.linalg.norm(final_pos)
        print(f"  Yaw {np.degrees(yaw_angle):.0f}°: Final position = ({final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}) m, Distance = {final_dist:.3f} m")
    
    plot_trajectory_comparison(trajectories, labels, "Comparison: Different Yaw Angles")

def compare_action_combinations():
    """Compare trajectories with different combinations of actions."""
    print("\nComparing different action combinations...")
    
    nozzle = Nozzle(length1=0.01, length2=0.01, length3=0.01, area=0.00009, mass=1.0)
    robot = Robot(dry_mass=1.0, init_length=0.3, init_width=0.15, 
                  max_contraction=0.06, nozzle=nozzle)
    robot.set_environment(density=1000)
    
    # Define different action combinations
    actions = [
        {'contraction': 0.06, 'coast_time': 1.0, 'yaw': 0.0, 'label': 'Max thrust, straight'},
        {'contraction': 0.03, 'coast_time': 1.0, 'yaw': 0.0, 'label': 'Half thrust, straight'},
        {'contraction': 0.06, 'coast_time': 0.5, 'yaw': 0.0, 'label': 'Max thrust, short coast'},
        {'contraction': 0.06, 'coast_time': 1.0, 'yaw': np.pi/6, 'label': 'Max thrust, turn right'},
        {'contraction': 0.06, 'coast_time': 1.0, 'yaw': -np.pi/6, 'label': 'Max thrust, turn left'},
    ]
    
    n_cycles = 5
    trajectories = []
    labels = []
    
    for action in actions:
        traj = simulate_trajectory(
            robot, n_cycles, 
            action['contraction'], 
            action['coast_time'], 
            action['yaw']
        )
        trajectories.append(traj)
        labels.append(action['label'])
        final_pos = traj['positions'][-1]
        print(f"  {action['label']}: Final position = ({final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}) m")
    
    plot_trajectory_comparison(trajectories, labels, "Comparison: Different Action Combinations")

def main():
    """Run all trajectory comparisons."""
    print("=" * 60)
    print("Robot Trajectory Comparison")
    print("=" * 60)
    
    # Compare individual action parameters
    # compare_contraction_levels()
    # compare_coast_times()
    compare_yaw_angles()
    
    # Compare action combinations
    # compare_action_combinations()

    # Test action/state comparison
    actions = np.array([
        [0.19323313, 0.29813224, 0.48714757],
        [7.7654147e-01, 3.8728118e-04, -8.1552941e-01],
        [0.98571205, 0.9917865, 0.99892616],
        [9.6167839e-01, 2.3841858e-07, -9.0644705e-01],
        [0.9982549, 0.01162207, 0.99545634],
        [1.097548e-01, 3.874302e-07, -9.995486e-01],
        [9.282575e-01, 9.834766e-07, -8.856592e-01],
        [0.9979527, 0.7998414, 0.9967793],
        [9.7881764e-01, 8.9406967e-08, -9.4052404e-01],
        [9.9689567e-01, 8.4903836e-04, 9.9000371e-01]
    ])
    
    states = np.array([
        [1.0155466e+00, 1.4098481e+00, 4.3595803e-04, -7.5057219e-03, -8.7840281e+00, -1.4406887e+00],
        [1.0133374, 1.4404503, 0.18948714, 0.09714148, -1.4771231, 12.405232],
        [9.8408753e-01, 1.4383022e+00, 5.6329452e-07, -2.0493079e-10, -2.2138830e+01, -5.8736938e-01],
        [0.9981343, 1.4710712, 0.15838267, 0.01597089, -10.856999, 12.6343],
        [1.0278028, 1.4743572, 0.05104179, -0.14053066, -14.307327, -10.275037],
        [1.0507892, 1.4647142, -0.07131741, -0.2293364, -17.46547, -0.5295754],
        [1.1534688, 1.3669854, 0.15083826, 0.02849274, -6.1852818, 12.59845],
        [1.1619021e+00, 1.3365263e+00, 2.0421480e-06, 6.5281060e-06, -2.5727917e+01, -7.1988845e-01],
        [1.1268356e+00, 1.3209202e+00, 1.6382785e-01, 4.4385507e-03, -1.4443778e+01, 1.2695701e+01],
        [1.0926594e+00, 1.3207027e+00, 1.6125831e-01, -1.3447195e-02, -1.6819508e+01, -1.2374420e+01]
    ])
    
    # Compare simulated trajectory with expected states
    comparison = compare_actions_with_states(actions, states, verbose=True)
    
    print("\n" + "=" * 60)
    print("All comparisons complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
