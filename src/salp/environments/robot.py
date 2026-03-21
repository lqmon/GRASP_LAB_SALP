from enum import Enum
import numpy as np


class Nozzle:
    """Represents a steerable nozzle for jet propulsion.
    
    Attributes:
        length1: First segment length of the nozzle
        length2: Second segment length of the nozzle
        length3: Third segment length of the nozzle
        area: Area of nozzle openning
        angle1: Rotation angle around y axis
        angle2: Rotation angle around z axis
        mass: Mass of the nozzle
        pos_com: Position of the nozzle center of mass
    """
    
    def __init__(self, length1: float = 0.0, length2: float = 0.0, 
                 length3: float = 0.0, area: float = 0.0, mass: float = 0.0):
        self.length1 = length1
        self.length2 = length2
        self.length3 = length3
        self.mass = mass    
        self.area = area
        self.angle1 = 0.0 
        self.angle2 = 0.0  
        self.yaw = 0.0  # yaw angle around z axis for control 
        self.gamma = np.pi / 4  # fixed tilt angle of nozzle downwards

        # Rotation matrices
        self.R_nm = None
        self.R_mb = None
        self.R_br = None
        
        # Initialize rotation matrices
        self._get_rotation_matrices()

    def set_angles(self, angle1: float, angle2: float):
        """Set the nozzle angles.
        
        Args:
            angle1: Rotation angle around y axis
            angle2: Rotation angle around z axis
        """
        self.angle1 = angle1
        self.angle2 = angle2
        self._get_rotation_matrices()
    
    def set_yaw_angle(self, yaw_angle: float):
        """Set the nozzle yaw angle (around z axis).
        
        Args:
            yaw_angle: Rotation angle around z axis
        """
        self.yaw = yaw_angle

    def solve_angles(self):
        """Inverse kinematics to find nozzle angles for a target direction.
        
        Args:
            target_direction: Desired 3D direction vector
        """ 
        target_direction = - np.array([np.cos(self.yaw), np.sin(self.yaw), 0])
        target_direction = self.R_br.transpose() @ target_direction

        val2 = np.clip(2*target_direction[2] - 1, -1.0, 1.0)
        self.angle2 = np.arccos(val2)
        if self.angle2 <= -np.pi:
            self.angle2 += 2*np.pi
        elif self.angle2 > np.pi:
            self.angle2 -= 2*np.pi

        if self.angle2 == 0:
            self.angle1 = 0.0
        else:
            a = 0.5 * (np.cos(self.angle2) - 1)
            b = np.sqrt(2) * np.sin(self.angle2) / 2
            c = target_direction[1]
            val1 = np.clip(c / np.sqrt(a**2 + b**2), -1.0, 1.0)
            self.angle1 = np.arcsin(val1) - np.arctan2(b, a)

        if self.angle1 <= -np.pi:
            self.angle1 += 2*np.pi
        elif self.angle1 > np.pi:
            self.angle1 -= 2*np.pi

        # print(f"Solved nozzle angles: angle1 = {self.angle1}, angle2 = {self.angle2}")
        
    def get_nozzle_position(self) -> np.ndarray:
        """Calculate the nozzle position in world frame.
        
        Returns:
            3D position vector of the nozzle tip
        """
        # Nozzle position in nozzle frame
        pos_x3 = self.length3 * np.cos(self.gamma)
        pos_y3 = 0
        pos_z3 = self.length3 * np.sin(self.gamma)
        nozzle_position = np.array([pos_x3, pos_y3, pos_z3]) 

        # Middle section tip position in body frame
        pos_x2 = 0
        pos_y2 = 0
        pos_z2 = self.length2
        middle_position = np.array([pos_x2, pos_y2, pos_z2])

        # Base section tip position in base frame
        pos_x1 = 0 
        pos_y1 = 0
        pos_z1 = self.length1
        base_position = np.array([pos_x1, pos_y1, pos_z1]) 

        position = self.R_br @ (base_position + self.R_mb @ (middle_position + self.R_nm @ nozzle_position))

        return position
    
    def get_nozzle_direction(self) -> np.ndarray:
        """Calculate the direction vector of the nozzle.
        
        Returns:
            3D direction unit vector in world frame
        """
        # Nozzle tilted at 45 degrees downwards
        
        pos_x = np.cos(self.gamma)
        pos_y = 0
        pos_z = np.sin(self.gamma)
        nozzle_direction = np.array([pos_x, pos_y, pos_z])

        direction = self.R_br @ self.R_mb @ self.R_nm @ nozzle_direction
        
        return direction
    
    def get_middle_position(self) -> np.ndarray:
        """Get the position of the second nozzle joint.
        
        Returns:
            3D position vector in body frame
        """
        pos_x = 0
        pos_y = 0
        pos_z = self.length1
        base_position = np.array([pos_x, pos_y, pos_z])

        # Middle section tip position in body frame
        pos_x2 = 0
        pos_y2 = 0
        pos_z2 = self.length2
        middle_position = np.array([pos_x2, pos_y2, pos_z2])

        position = self.R_br @ (base_position + self.R_mb @ middle_position)
        # print("Middle position:", position)
        return position

    def _get_rotation_matrices(self) -> np.ndarray:
        """Calculate the overall rotation matrix of the nozzle.
        
        Returns:
            3x3 rotation matrix
        """
        R_theta_fixed = np.array([[np.cos(self.gamma), 0, -np.sin(self.gamma)],
                                     [0, 1, 0],
                                     [np.sin(self.gamma), 0, np.cos(self.gamma)]])
        
        R_nozzle = np.array([[np.cos(self.angle2), -np.sin(self.angle2), 0],
                             [np.sin(self.angle2), np.cos(self.angle2), 0],
                             [0, 0, 1]])
        
        R_middle = np.array([[np.cos(self.angle1), -np.sin(self.angle1), 0],
                             [np.sin(self.angle1), np.cos(self.angle1), 0],
                             [0, 0, 1]])

        # Convert from nozzle frame to body frame
        R_base = np.array([[0, 0, -1],
                           [0, 1, 0],
                           [1, 0, 0]])

        self.R_nm = R_theta_fixed @ R_nozzle
        self.R_mb = R_middle
        self.R_br = R_base
        
         
class Robot:
    """Simulates a jet-propelled robot with deformable body.
    
    The robot uses water jet propulsion and can contract/expand its body.
    Supports different phases: REFILL, JET, COAST, and REST.
    """
    
    class Phase(Enum):
        REFILL = 0
        JET = 1
        COAST = 2
        REST = 3

    phase = [Phase.REFILL, Phase.JET, Phase.COAST, Phase.REST]

    def __init__(self, dry_mass: float, init_length: float, init_width: float,
                 max_contraction: float, nozzle: Nozzle):
        """Initialize the robot.
        
        Args:
            dry_mass: Mass of the robot without water (kg)
            init_length: Initial length of the robot (m)
            init_width: Initial width of the robot (m)
            max_contraction: Maximum contraction distance (m)
            nozzle: Nozzle object for jet propulsion
        """

        # constant properties during cycle
        self.dry_mass = dry_mass  # kg
        self.init_length = init_length  # meters
        self.init_width = init_width  # meters
        self.max_contraction = max_contraction  # max contraction length
        self.density = 1000  # kg/m^3, density of water
        self.dt = 0.01  # time step
        self.nozzle = nozzle
        self.refill_time = 0.0
        self.jet_time = 0.0
        self.coast_time = 0.0
        self.contraction = 0.0  # contraction level
        self._contract_rate = 0.0
        self._release_rateS = 0.0
        self._drag_coefficents = [0.4, 1.0] # min and max drag coefficients for different shapes

        # cycle tracking 
        self.state = self.phase[3]  # initial state is rest
        self.cycle = 0
        self.time = 0.0
        self.cycle_time = 0.0

        # properties updated each step
        self.length = 0.0
        self.width = 0.0
        self.area = 0.0
        self.volume = 0.0  # water volume inside the robot
        self.mass = np.diag(np.zeros(3))  # total mass including water
        self.water_mass = 0.0
        self.prev_water_volume = 0.0
        self.prev_water_mass = 0.0
        self.jet_velocity = np.zeros(3)  # jet velocity vector
        self.jet_force = np.zeros(3)  # jet force vector
        self.jet_torque = np.zeros(3)  # jet torque vector
        self.drag_coefficient = 0.0
        self.drag_force = np.zeros(3)  # drag force vector
        self.drag_torque = np.zeros(3)  # drag torque vector

        # State variables
        self.position = np.zeros(3)  # x, y, z positions
        self.velocity = np.zeros(3)  # x, y, z velocities
        self.velocity_world = np.zeros(3)  # x, y, z velocities in world frame
        self.acceleration = np.zeros(3)  # x, y, z accelerations

        self.euler_angle = np.zeros(3)  # roll, pitch, yaw
        self.euler_angle_rate = np.zeros(3)  # roll, pitch, yaw rates
        self.angular_velocity = np.zeros(3) # body frame angular velocity 
        self.angular_acceleration = np.zeros(3)  # body frame angular acceleration

        # record cycle history
        self.state_history = []
        self.position_history = []
        self.velocity_history = []
        self.acceleration_history = []
        self.euler_angle_history = []
        self.euler_angle_rate_history = []
        self.angular_velocity_history = []
        self.angular_acceleration_history = []
        self.length_history = []
        self.width_history = []
        self.area_history = []
        self.volume_history = []
        self.mass_history = []
        self.jet_velocity_history = []
        self.jet_force_history = []
        self.jet_torque_history = []
        self.drag_coefficient_history = []
        self.drag_force_history = []
        self.drag_torque_history = []


    def set_environment(self, density: float):
        """Set the environment properties.
        
        Args:
            density: Fluid density (kg/m^3)
        """
        self.density = density

    def reset(self):
        """Reset the robot to initial state."""
        self.time = 0.0
        self.cycle_time = 0.0
        self.cycle = 0
        self.state = self.phase[3]  # rest state 

        self.position = np.zeros(3)  # x, y, z positions
        self.velocity = np.zeros(3)  # x, y, z velocities
        self.velocity_world = np.zeros(3)  # x, y, z velocities in world frame
        self.acceleration = np.zeros(3)  # x, y, z accelerations
        self.euler_angle = np.zeros(3)  # roll, pitch, yaw
        self.euler_angle_rate = np.zeros(3)  # roll, pitch, yaw rates
        self.angular_velocity = np.zeros(3)  # yaw rate
        self.angular_acceleration = np.zeros(3)  # body frame angular acceleration      

        self.length = self.init_length
        self.width = self.init_width
        self.area = self._get_cross_sectional_area()
        self.volume = self._get_water_volume()
        self.water_mass = self._get_water_mass()
        self.mass = self.get_mass()
        self.prev_water_mass = self.mass
        self.prev_water_volume = self.volume
        self.prev_I = self.get_inertia_matrix()
        self.drag_coefficient = self._get_drag_coefficient()

        # empty cycle history
        self.state_history = []
        self.position_history = []
        self.velocity_history = []
        self.acceleration_history = []
        self.euler_angle_history = []
        self.euler_angle_rate_history = []
        self.angular_velocity_history = []
        self.angular_acceleration_history = []
        self.length_history = []
        self.width_history = []
        self.area_history = []
        self.volume_history = []
        self.mass_history = []
        self.jet_velocity_history = []
        self.jet_force_history = []
        self.jet_torque_history = []
        self.drag_coefficient_history = []
        self.drag_force_history = []
        self.drag_torque_history = []

    def set_control(self, contraction: float, coast_time: float, nozzle_angles: np.ndarray):
        """Set control inputs for the robot.
        
        Args:
            contraction: Desired contraction distance (m)
            coast_time: Duration of coast phase (s)
            angle: Nozzle steering angle
        """

        # set control inputs    
        self.contraction = contraction
        self.coast_time = coast_time
        self.nozzle.set_angles(angle1=nozzle_angles[0], angle2=nozzle_angles[1])

        # proceed to next cycle
        self.cycle += 1
        self.cycle_time = 0.0

        # if self.cycle % 1000 == 0:
        #     print(f"Cycle {self.cycle}")

        self.refill_time = self._contract_model()
        self.jet_time = self._release_model()
        # print(f"Refill time = {self.refill_time:.2f}s, Jet time = {self.jet_time:.2f}s")

    def update_state(self):
        """Determine current phase based on cycle time.
        
        Returns:
            Current phase state
        """
        if self.cycle_time <= self.refill_time:
            self.state = self.phase[0]  # contract
        elif self.cycle_time <= self.refill_time + self.jet_time:
            self.state = self.phase[1]  # release
        elif self.cycle_time <= self.refill_time + self.jet_time + self.coast_time:
            self.state = self.phase[2]  # coast
        else:
            self.state = self.phase[3]  # reset to rest
    
    def update_properties(self):
        """Update robot properties based on current state."""
        self.prev_water_volume = self.volume
        self.prev_water_mass = self.prev_water_volume * self.density

        self.length = self.get_current_length()
        self.width = self.get_current_width()
        self.area = self._get_cross_sectional_area()
        self.volume = self._get_water_volume()
        self.mass = self.get_mass()[0,0]
        self.drag_coefficient = self._get_drag_coefficient()

    def step(self):
        """Advance simulation by one time step."""
        
        # proceed to next time step
        self.cycle_time += self.dt
        self.time += self.dt
        self.update_state()
        self.update_properties()
        self.update_dynamics()

    # Define getter functions for current values
    def get_current_values(self):
        return {
            'state_history': self.state,
            'position_history': self.position.copy(),
            'velocity_history': self.velocity.copy(),
            'acceleration_history': self.acceleration.copy(),
            'euler_angle_history': self.euler_angle.copy(),
            'euler_angle_rate_history': self.euler_angle_rate.copy(),
            'angular_velocity_history': self.angular_velocity.copy(),
            'angular_acceleration_history': self.angular_acceleration.copy(),
            'length_history': self.length,
            'width_history': self.width,
            'area_history': self.area,
            'volume_history': self.volume,
            'mass_history': self.mass[0,0],  # store only scalar mass value
            'jet_velocity_history': self.jet_velocity,
            'jet_force_history': self.jet_force,
            'jet_torque_history': self.jet_torque.copy(),
            'drag_coefficient_history': self.drag_coefficient,
            'drag_force_history': self.drag_force,
            'drag_torque_history': self.drag_torque.copy(),
            'nozzle_yaw_history': self.nozzle.yaw,
        }

    def step_through_cycle(self):
        """Step through an entire breathing cycle and collect state history.
        
        Returns:
            Tuple of arrays containing time history of various state variables
        """

        total_cycle_time = self.refill_time + self.jet_time + self.coast_time

        # Initialize history lists with current values
        for attr_name, initial_value in self.get_current_values().items():
            setattr(self, attr_name, [initial_value])

        while self.cycle_time < total_cycle_time:
            self.step()
            
            # Append current values to history lists
            for attr_name, current_value in self.get_current_values().items():
                getattr(self, attr_name).append(current_value)

        # Convert histories to numpy arrays for easier handling
        history_names = self.get_current_values().keys()
        for attr_name in history_names:
            setattr(self, attr_name, np.array(getattr(self, attr_name))) 

    def _to_euler_angle_rate(self) -> np.ndarray:
        """Convert angular velocity to Euler angle rates.
        
        Returns:
            Euler angle rate vector
        """
        phi, theta, psi = self.euler_angle

        T = np.array([[1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
                      [0, np.cos(phi), -np.sin(phi)],
                      [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]])

        return T @ self.angular_velocity

    def _to_world_frame(self, vector: np.ndarray) -> np.ndarray:
        """Convert a vector from body frame to world frame.
        
        Args:
            vector: 3D vector in body frame
            
        Returns:
            3D vector in world frame
        """
        phi, theta, psi = self.euler_angle

        R_x = np.array([[1, 0, 0],
                        [0, np.cos(phi), -np.sin(phi)],
                        [0, np.sin(phi), np.cos(phi)]])
        
        R_y = np.array([[np.cos(theta), 0, np.sin(theta)],
                        [0, 1, 0],
                        [-np.sin(theta), 0, np.cos(theta)]])
        
        R_z = np.array([[np.cos(psi), -np.sin(psi), 0],
                        [np.sin(psi), np.cos(psi), 0],
                        [0, 0, 1]])
        
        R = R_z @ R_y @ R_x

        return R @ vector

    def update_dynamics(self):

        self.acceleration = self._newton_equations()
        self.angular_acceleration = self._euler_equations()
        self._update_motion_states()

    def _newton_equations(self) -> np.ndarray:
        """Compute translational accelerations using Newton's equations.
        
        Returns:
            3D acceleration vector
        """
        F_coriolis = self._get_coriolis_force()
        self.drag_force = self._get_drag_force()
        self.jet_force = self._get_jet_force()
        self.mass = self.get_mass()

        return np.linalg.inv(self.mass) @ (self.jet_force + self.drag_force + F_coriolis)

    def _euler_equations(self) -> np.ndarray:
        """Compute angular accelerations using Euler's equations.
        
        Returns:
            3D angular acceleration vector
        """
        T_asymmetry = np.array([0.0, 0.0, 0.1*np.linalg.norm(self.velocity)])  # TODO: Implement asymmetry torque
        T_coriolis = self._get_coriolis_torque()
        self.drag_torque = self._get_drag_torque()
        self.jet_torque = self._get_jet_torque()
        T_deform = self.get_inertia_matrix_rate() @ self.angular_velocity
        # T_jet = np.array([0.0, 0.0, 0.0])  # test orientation

        I = self.get_inertia_matrix()

        return np.linalg.inv(I) @ (self.jet_torque + self.drag_torque + T_coriolis + T_asymmetry - T_deform)

    def _update_motion_states(self):
        """Update robot state variables based on accelerations."""
        self.velocity += self.acceleration * self.dt
        self.angular_velocity += self.angular_acceleration * self.dt

        self.euler_angle_rate = self._to_euler_angle_rate()
        self.euler_angle += self.euler_angle_rate * self.dt
        self.velocity_world = self._to_world_frame(self.velocity)
        self.position += self.velocity_world * self.dt

    def get_inertia_matrix(self) -> np.ndarray:
        """Calculate moment of inertia matrix.
        
        Note: Currently only considers water inertia.
        
        Returns:
            3x3 inertia matrix
        """
        r = self._get_jet_moment_arm()
        I_nozzle = self.nozzle.mass * np.linalg.norm(r) ** 2 * np.diag(np.array([0, 1, 1]))

        I_xx = 0.2 * self.mass[0][0] * ((self.width/2) ** 2 + (self.width/2) ** 2)
        I_yy = 0.2 * self.mass[0][0] * ((self.length/2) ** 2 + (self.width/2) ** 2)
        I_zz = 0.2 * self.mass[0][0] * ((self.width/2) ** 2 + (self.length/2) ** 2)

        I_robot = np.diag([I_xx, I_yy, I_zz])

        return I_robot + I_nozzle
    
    def get_inertia_matrix_rate(self) -> np.ndarray:
        """Calculate rate of change of inertia matrix.
        
        Note: Currently not implemented.
        
        Returns:
            3x3 inertia matrix rate (zeros)
        """

        I_rate = (self.get_inertia_matrix() - self.prev_I) / self.dt
        self.prev_I = self.get_inertia_matrix()

        return I_rate

    def _get_jet_moment_arm(self) -> np.ndarray:
        """Calculate moment arm for jet force.
        
        Returns:
            3D moment arm vector
        """
        r_nozzle = self.nozzle.get_middle_position()
        r_robot = np.array([-self.length/2, 0.0, 0.0])  # center of mass at origin
        return r_nozzle + r_robot
    
    def _get_jet_torque(self) -> np.ndarray:
        """Calculate torque from jet force.
        
        Returns:
            3D torque vector
        """

        return np.cross(self._get_jet_moment_arm(), self.jet_force)
    
    def _get_jet_force(self) -> np.ndarray:
        """Calculate jet propulsion force.
        
        Returns:
            3D force vector
        """
        self.jet_velocity = self._get_jet_velocity()
        if self.state != self.phase[1]:  # only produce jet force during release phase
            return np.zeros(3)

        mass_rate = (self.water_mass - self.prev_water_mass) / self.dt
        C_discharge = 0.1  # discharge coefficient

        return C_discharge * mass_rate * self.jet_velocity
    
    def _get_jet_velocity(self) -> np.ndarray:
        """Calculate jet velocity vector.
        
        Returns:
            3D velocity vector in robot frame
        """
        if self.state != self.phase[1]:  # only produce jet velocity during release phase
            return np.zeros(3)      

        volume_rate = -(self.volume - self.prev_water_volume) / self.dt
        jet_speed = volume_rate / self.nozzle.area
        direction = self.nozzle.get_nozzle_direction()

        return direction * jet_speed
    
    def _length_width_relation(self, length: float) -> float:
        """Calculate width based on length (volume conservation).
        
        Args:
            length: Current body length
            
        Returns:
            Corresponding body width
        """
        return self.init_length - length + self.init_width

    def _get_drag_coefficient(self) -> float:
        """Calculate drag coefficient based on body shape.
        
        More elongated (contracted) = lower drag, more spherical = higher drag.
        
        Returns:
            Drag coefficient
        """

        aspect_ratio = self.length / self.width
        
        init_aspect_ratio = self.init_length / self.init_width  # most elongated
        contracted_length = self.init_length - self.max_contraction
        contracted_width = self._length_width_relation(contracted_length)
        min_aspect_ratio = contracted_length / contracted_width  # most spherical
        
        # Normalize to [0, 1]: 0 = most spherical, 1 = most elongated
        normalized_ratio = (aspect_ratio - min_aspect_ratio) / (init_aspect_ratio - min_aspect_ratio)
        normalized_ratio = np.clip(normalized_ratio, 0, 1)
        
        C_d = self._drag_coefficents[1] - normalized_ratio * (self._drag_coefficents[1] - self._drag_coefficents[0])

        return C_d

    def _get_drag_torque(self) -> np.ndarray:
        """Calculate drag torque on the robot.
        
        Returns:
            3D torque vector
        """
        T_drag = - self.density * self.drag_coefficient * (self.width / 2) * \
            (self.length / 2) ** 4 * np.linalg.norm(self.angular_velocity) * self.angular_velocity
        self.drag_torque = T_drag
        return T_drag
    
    def _get_drag_force(self) -> np.ndarray:
        """Calculate drag force on the robot.
        
        Returns:
            3D force vector
        """
        #TODO: drag has a slight discountinuity!
        self.drag_coefficient = self._get_drag_coefficient()
        F_drag = -0.5 * self.density * self.area * self.drag_coefficient * np.linalg.norm(self.velocity) * self.velocity

        F_linear = -0.5 * self.density * self.area * self.drag_coefficient * self.velocity

        return F_drag + F_linear
    
    def _get_added_mass(self) -> float:
        """Calculate added mass from surrounding fluid.
        
        Returns:
            Added mass (currently not implemented)
        """
        # TODO: Implement added mass calculation
        # added_mass = 0.5 * self.density * self._get_water_volume()
        return 0.0
    
    def _get_coriolis_force(self) -> np.ndarray:
        """Calculate Coriolis force.
        
        Returns:
            3D force vector
        """
        return self.get_mass() @ np.cross(self.angular_velocity, self.velocity)

    def _get_coriolis_torque(self) -> np.ndarray:
        """Calculate Coriolis torque.
        
        Returns:
            3D torque vector
        """
        return -np.cross(self.angular_velocity, self.get_inertia_matrix() @ self.angular_velocity)

    def get_current_length(self) -> float:
        """Calculate current body length based on phase.
        
        Returns:
            Current length in meters
        """
        if self.state == self.phase[0]:  # inhale
            length = self.init_length - self.cycle_time*self._contract_rate
        elif self.state == self.phase[1]:  # exhale
            length = self.init_length - self.contraction + (self.cycle_time - self.refill_time)*self._release_rate
        else:
            length = self.init_length

        return length
    
    def get_current_width(self) -> float:
        """Calculate current body width based on phase.
        
        Returns:
            Current width in meters
        """
        if self.state == self.phase[0]:  # inhale
            width = self.init_width + self.cycle_time*self._contract_rate
        elif self.state == self.phase[1]:  # exhale
            width = self.init_width + self.contraction - (self.cycle_time - self.refill_time)*self._release_rate
        else:
            width = self.init_width

        return width

    def _get_cross_sectional_area(self) -> float:

        area = np.pi * (self.length/2) * (self.width/2)
        return area

    def _get_water_volume(self) -> float:
        
        volume = 4/3 * np.pi * (self.length/2) * (self.width/2) ** 2

        return volume

    def _get_water_mass(self) -> float:
        
        water_mass = self.density * self._get_water_volume()   

        return water_mass

    def get_mass(self) -> float:
        """Calculate total mass including water.
        
        Returns:
            Mass matrix (diagonal)
        """
        self.water_mass = self._get_water_mass()
        mass = self.dry_mass + self.water_mass + self.nozzle.mass
        mass = mass * np.diag(np.ones(3))

        return mass

    def _contract_model(self) -> float:
        """Calculate contraction time based on contraction distance.
        
        Returns:
            Time duration in seconds
        """
        self._contract_rate = 0.06/3  # m/s
        return self.contraction / self._contract_rate

    def _release_model(self) -> float:
        """Calculate release time based on contraction distance.
        
        Returns:
            Time duration in seconds
        """
        self._release_rate = 0.06/1.5  # m/s
        return self.contraction / self._release_rate


if __name__ == "__main__":
    from plotting import (
        plot_angular_velocity, plot_drag_torque, plot_angular_acceleration,
        plot_euler_angles, plot_robot_geometry, plot_robot_mass, plot_mass_rate,
        plot_volume_rate, plot_cross_sectional_area, plot_jet_velocity,
        plot_jet_properties, plot_drag_coefficient, plot_drag_properties,
        plot_robot_position, plot_robot_velocity, plot_jet_torque, plot_trajectory_xy,
        plot_nozzle_direction
    )

    # Test the Robot and Nozzle classes
    nozzle = Nozzle(length1=0.05, length2=0.05, length3=0.05, area=0.00016, mass=1.0)
    robot = Robot(dry_mass=1.0, init_length=0.3, init_width=0.15, 
                  max_contraction=0.06, nozzle=nozzle)
    robot.nozzle.set_angles(angle1=0.0, angle2=np.pi)  # set nozzle angles
    
    robot.set_environment(density=1000)  # water density in kg/m^3
    robot.reset()
    
    # Step through multiple cycles and collect state data
    n_cycles = 1
    
    # Initialize accumulators for all cycle data
    all_time_data = []
    all_state_data = []
    all_position_data = []
    all_velocity_data = []
    all_acceleration_data = []
    all_euler_angle_data = []
    all_euler_angle_rate_data = []
    all_angular_velocity_data = []
    all_angular_acceleration_data = []
    all_length_data = []
    all_width_data = []
    all_area_data = []
    all_volume_data = []
    all_mass_data = []
    all_jet_velocity_data = []
    all_jet_force_data = []
    all_jet_torque_data = []
    all_drag_coefficient_data = []
    all_drag_force_data = []
    all_drag_torque_data = []

    # Test action sequence
    test_actions = np.array([
        [0.22891083, 0.06766406, -0.9850989],
        [0.2842933, 0.963629, 0.9741967],
        [0.8862339, 0.32421368, -0.9328714],
        [0.05561769, 0.91966885, 0.85212207],
        # [0.25341812, 0.6691348, 0.7325938],
        # [0.8321035, 0.23156995, -0.92043316],
        # [0.05115855, 0.96011114, 0.8534517],
        # [0.24099252, 0.71873295, 0.7108506],
        # [0.6869794, 0.04822099, -0.86440706],
        # [0.2337848, 0.9906088, 0.99013484],
        # [0.6945975, 0.09169137, -0.8268971],
        # [0.06978488, 0.9933312, 0.9369149],
        # [0.06852683, 0.7448164, 0.8688922],
        # [0.4922041, 0.10394484, 0.8375015],
        # [0.68481743, 0.00496772, -0.99800324],
        # [0.9857271, 0.647208, 0.99998224],
        # [0.7742654, 0.83240354, -0.66497386],
        # [0.78678393, 0.01270097, 0.9582412],
        # [3.6354065e-03, 1.2317300e-04, -9.9999970e-01],
        # [6.2082112e-03, 3.0893087e-04, -9.9999964e-01],
        # [8.4684789e-03, 2.1675229e-04, -9.9999875e-01],
        # [1.7061472e-02, 3.0627847e-04, -9.9999666e-01],
        # [6.9575727e-02, 5.1766634e-04, -9.9997485e-01],
        # [0.6683986, 0.02849919, -0.984029],
        # [0.99468315, 0.3537972, 0.9998045],
        # [0.6911941, 0.04722786, -0.96879447],
        # [0.9897277, 0.33548862, 0.9979936],
        # [0.89258796, 0.00411212, -0.96228147],
    ])

    for i in range(len(test_actions)):
        # robot.nozzle.set_yaw_angle(yaw_angle=np.random.uniform(-np.pi/2, np.pi/2))
        # contraction = np.random.uniform(0.0, 0.06)
        # coast_time = np.random.uniform(0.0, 2.0)
        # TODO: debug this Action taken: Inhale: 0.51, Coast Time: 0.86, Nozzle Yaw: -0.85 rad

        contraction = 0.06 * test_actions[i, 0]
        coast_time = 10 * test_actions[i, 1]
        yaw_angle = test_actions[i, 2] * np.pi/2    

        robot.nozzle.set_yaw_angle(yaw_angle=yaw_angle)

        robot.nozzle.solve_angles()
        robot.set_control(contraction=contraction, coast_time=coast_time, nozzle_angles=np.array([robot.nozzle.angle1, robot.nozzle.angle2]))
        robot.nozzle.set_angles(angle1=-0.43, angle2=1.14)
        plot_nozzle_direction(robot.nozzle, title=f'Cycle {i+1} Nozzle Direction')
        robot.step_through_cycle()
    
        # Create time array for this cycle
        cycle_start_time = robot.time - robot.cycle_time
        time_array = np.arange(cycle_start_time, robot.time + robot.dt, robot.dt)[:len(robot.length_history)]
        
        # Accumulate data from each cycle
        all_time_data.extend(time_array)
        all_state_data.extend(robot.state_history)
        all_position_data.extend(robot.position_history)
        all_velocity_data.extend(robot.velocity_history)
        all_acceleration_data.extend(robot.acceleration_history)
        all_euler_angle_data.extend(robot.euler_angle_history)
        all_euler_angle_rate_data.extend(robot.euler_angle_rate_history)
        all_angular_velocity_data.extend(robot.angular_velocity_history)
        all_angular_acceleration_data.extend(robot.angular_acceleration_history)
        all_length_data.extend(robot.length_history)
        all_width_data.extend(robot.width_history)
        all_area_data.extend(robot.area_history)
        all_volume_data.extend(robot.volume_history)
        all_mass_data.extend(robot.mass_history)
        all_jet_velocity_data.extend(robot.jet_velocity_history)
        all_jet_force_data.extend(robot.jet_force_history)
        all_jet_torque_data.extend(robot.jet_torque_history)
        all_drag_coefficient_data.extend(robot.drag_coefficient_history)
        all_drag_force_data.extend(robot.drag_force_history)
        all_drag_torque_data.extend(robot.drag_torque_history)
    
    # Convert accumulated data to numpy arrays
    all_time_data = np.array(all_time_data)
    all_state_data = np.array(all_state_data)
    all_position_data = np.array(all_position_data)
    all_velocity_data = np.array(all_velocity_data)
    all_acceleration_data = np.array(all_acceleration_data)
    all_euler_angle_data = np.array(all_euler_angle_data)
    all_euler_angle_rate_data = np.array(all_euler_angle_rate_data)
    all_angular_velocity_data = np.array(all_angular_velocity_data)
    all_angular_acceleration_data = np.array(all_angular_acceleration_data)
    all_length_data = np.array(all_length_data)
    all_width_data = np.array(all_width_data)
    all_area_data = np.array(all_area_data)
    all_volume_data = np.array(all_volume_data)
    all_mass_data = np.array(all_mass_data)
    all_jet_velocity_data = np.array(all_jet_velocity_data)
    all_jet_force_data = np.array(all_jet_force_data)
    all_jet_torque_data = np.array(all_jet_torque_data)
    all_drag_coefficient_data = np.array(all_drag_coefficient_data)
    all_drag_force_data = np.array(all_drag_force_data)
    all_drag_torque_data = np.array(all_drag_torque_data)
    
    # Plot all cycles together
    # plot_robot_geometry(all_time_data, all_length_data, all_width_data, all_state_data)
    # plot_cross_sectional_area(all_time_data, all_area_data, all_state_data)  
    # plot_robot_mass(all_time_data, all_mass_data, all_state_data) 
    # plot_volume_rate(all_time_data, all_volume_data, all_state_data)   
    # plot_mass_rate(all_time_data, all_mass_data, all_state_data)
    # plot_jet_velocity(all_time_data, all_jet_velocity_data, all_state_data)
    # plot_jet_properties(all_time_data, all_jet_force_data, all_state_data)
    # plot_drag_coefficient(all_time_data, all_drag_coefficient_data, all_state_data)
    # plot_drag_properties(all_time_data, all_drag_force_data, all_state_data)
    # plot_robot_velocity(all_time_data, all_velocity_data, all_state_data)  
    # plot_robot_position(all_time_data, all_position_data, all_state_data)
    # plot_angular_velocity(all_time_data, all_angular_velocity_data, all_state_data)
    # plot_jet_torque(all_time_data, all_jet_torque_data, all_state_data)
    # plot_drag_torque(all_time_data, all_drag_torque_data, all_state_data)
    # plot_angular_acceleration(all_time_data, all_angular_acceleration_data, all_state_data)
    # plot_euler_angles(all_time_data, all_euler_angle_data, all_state_data)
    
    # Plot trajectory in x-y plane with yaw orientation
    plot_trajectory_xy(all_position_data, all_state_data, all_euler_angle_data)
    
