"""
SALP Robot Simulation  –  Chain of 5 with Deep Q Learning
===================================================
Five salps linked in a linear chain by rigid-stick connectors.
Autonomous learning using Deep Q Networks to reach goal positions.
Reward: -1 per time step + bonuses for reaching goal.

Controls (Manual Mode)
--------
  Click any salp to select it (glowing ring shows selection)
  HOLD W : inhale    A / D : steer nozzle
  Keys 1-5 : select salp by number
  Space : Toggle autonomous/manual control
  ESC   : quit
"""

import math
import random
import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
#  DEEP Q NETWORK
# ─────────────────────────────────────────────────────────────────────────────

class DQNNetwork(nn.Module):
    """Deep Q Network for autonomous salp control."""
    
    def __init__(self, state_size: int = 8, action_size: int = 9, hidden_size: int = 128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    """Deep Q Learning Agent for SALP control.
    
    Controls the leading salp (salp 0) while being aware of all salps' positions.
    State includes all salps' distances to enable cooperative goal-reaching.
    Reward based on minimum distance (closest salp to goal).
    """
    
    def __init__(
        self,
        state_size: int = 22,  # Updated for velocity awareness: 7 base + 5 distances + 10 velocities
        action_size: int = 9,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        memory_size: int = 10000,
        batch_size: int = 64,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay memory
        self.memory = deque(maxlen=memory_size)
        self.steps = 0
    
    def get_state(self, salps, goal_pos, env_width, env_height):
        """Extract state features from all salps and goal.
        
        State includes:
        - Leading salp (index 0): position, velocity, angle
        - Goal position
        - Distance of each salp to goal (for cooperative awareness)
        - Velocities of ALL salps (for movement estimation)
        
        Total: 12D for leading salp + 2D goal + 5D distances + 10D velocities = 29 dimensions
        Allows agent to predict chain movement MORE ACCURATELY by observing all velocities.
        """
        salp = salps[0]
        
        # Leading salp (salp 0): position and velocity
        norm_x = salp.pos[0] / env_width
        norm_y = salp.pos[1] / env_height
        norm_vx = salp.vel[0] / 10.0
        norm_vy = salp.vel[1] / 10.0
        
        # Goal position
        norm_goal_x = goal_pos[0] / env_width
        norm_goal_y = goal_pos[1] / env_height
        
        state_list = [
            norm_x, norm_y,
            norm_vx, norm_vy,
            salp.angle / math.pi,
            norm_goal_x, norm_goal_y,
        ]
        
        # Add distances for ALL salps (enables cooperative awareness)
        # Agent learns to optimize for closest salp reaching goal
        for s in salps:
            dist_to_goal = float(np.linalg.norm(s.pos - goal_pos))
            state_list.append(dist_to_goal / 500.0)
        
        # Add velocities for ALL salps (enables movement estimation)
        # Agent can predict how chain will move based on current velocities
        for s in salps:
            vx = s.vel[0] / 10.0
            vy = s.vel[1] / 10.0
            state_list.append(vx)
            state_list.append(vy)
        
        state = np.array(state_list, dtype=np.float32)
        return state
    
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax(dim=1).item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return None
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q-values
        next_q_values = self.target_network(next_states).max(dim=1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Loss
        loss = F.mse_loss(q_values, target_q_values.detach())
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train_mode(self):
        self.q_network.train()
    
    def eval_mode(self):
        self.q_network.eval()



#  SALP  –  one robot instance
# ─────────────────────────────────────────────────────────────────────────────

class Salp:
    def __init__(
        self,
        radius:        float = 30,
        start_pos:     tuple = (400, 300),
        max_thrust:    float = 100,
        drag:          float = 0.98,
        inhale_frames: int   = 120,
        exhale_frames: int   = 150,
    ):
        self.radius        = radius
        self.start_pos     = np.array(start_pos, dtype=float)
        self.max_thrust    = max_thrust
        self.drag          = drag
        self.inhale_frames = inhale_frames
        self.exhale_frames = exhale_frames

        self.max_nozzle_angle  = math.pi / 3
        self.nozzle_move_speed = 0.05

        self.reset()

    def reset(self):
        self.pos          = self.start_pos.copy()
        self.vel          = np.array([0.0, 0.0])
        self.angle        = 0.0
        self.angular_vel  = 0.0

        self.nozzle_angle  = 0.0
        self.target_nozzle = 0.0

        self.phase           = "rest"
        self.phase_timer     = 0
        self.water           = 0.0
        self.exhale_duration = self.exhale_frames

        self.semi_a = self.radius * 1.3
        self.semi_b = self.radius * 0.8

        self.inhale_held  = False
        self.nozzle_input = 0.0

    @property
    def speed(self):
        return float(np.linalg.norm(self.vel))

    @property
    def max_extent(self):
        return max(self.semi_a, self.semi_b)


# ─────────────────────────────────────────────────────────────────────────────
#  SALP ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────

class SalpEnv:

    COLOURS = [
        (100, 140, 200),   # blue
        (100, 190, 160),   # teal
        (190, 140, 100),   # sand
        (180, 100, 160),   # rose
        (160, 100, 190),   # purple
    ]

    # Stick connector appearance
    STICK_LEN    = 28    # px of the rigid rod between salp surfaces
    STICK_WIDTH  = 4
    STICK_COLOUR = (160, 180, 200)
    JOINT_RADIUS = 5
    JOINT_COLOUR = (220, 230, 240)

    # Spring stiffness for the chain constraint (0–1, applied per frame)
    # Increased to 1.0 for RIGID formation - salps maintain spawned line perfectly
    SPRING_K = 1.0

    def __init__(
        self,
        salps:  list,
        width:  int = 1600,
        height: int = 900,
        margin: int = 60,
        fps:    int = 60,
        thinking_frames: int = 1,  # CRITICAL: Decision every 1 frame for responsive control
    ):
        self.salps  = salps
        self.width  = width
        self.height = height
        self.margin = margin
        self.fps    = fps
        self.thinking_frames = thinking_frames  # Frames between agent decisions
        self.thinking_counter = 0  # Tracks frames since last decision

        # FORMATION POLE SYSTEM: All salps in fixed rigid formation around moving pole
        # Calculate pole center from salp spawn positions
        pole_x = sum(s.start_pos[0] for s in salps) / len(salps)
        pole_y = sum(s.start_pos[1] for s in salps) / len(salps)
        self.pole_pos = np.array([pole_x, pole_y], dtype=float)
        self.pole_vel = np.array([0.0, 0.0], dtype=float)  # Pole moves with formation
        
        # Pre-compute distance and angle from each salp to pole (FIXED relative positions)
        # Salps stay parallel: same distance/angle relationship at all times
        self.pole_distances = []
        self.pole_angles = []
        for s in salps:
            delta = s.start_pos - self.pole_pos
            dist = float(np.linalg.norm(delta))
            angle = math.atan2(delta[1], delta[0])
            self.pole_distances.append(dist)
            self.pole_angles.append(angle)
        
        # POLE SPRING STIFFNESS: How tightly salps stick to formation around pole
        # 1.0 = absolutely rigid formation, zero drift
        self.POLE_SPRING_K = 1.0
        
        # POLE DRAG: Pole decelerates smoothly like a large object in water
        self.POLE_DRAG = 0.96

        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("SALP Chain Simulation - Deep Q Learning")
        self.clock  = pygame.time.Clock()
        self.font   = pygame.font.Font(None, 22)
        self.small  = pygame.font.Font(None, 17)

        self._flash_timer    = 0
        self._flash_duration = 18
        self.selected        = 0   # index of currently selected salp

        # Random gold star goal
        inner = margin + 60
        self.goal_pos = np.array([
            random.uniform(inner, width  - inner),
            random.uniform(inner, height - inner),
        ])
        
        # Autonomous control
        self.autonomous_mode = True
        self.episode_steps = 0
        self.max_episode_steps = 500
        self.total_reward = 0.0
        self.last_distance = 0.0

    # ─────────────────────────────────────────────────────────────────────
    #  Main step
    # ─────────────────────────────────────────────────────────────────────

    def step(self):
        # Update individual salp kinematics
        for salp in self.salps:
            self._update_nozzle(salp)
            self._update_breathing(salp)
            self._update_physics(salp)

        # CALCULATE FORMATION MOVEMENT: Average velocity of all salps
        # This represents the collective thrust moving the entire formation
        avg_vel = np.array([0.0, 0.0])
        for salp in self.salps:
            avg_vel += salp.vel
        avg_vel /= len(self.salps)
        
        # MOVE POLE with formation: Pole inherits the collective motion
        self.pole_vel = avg_vel * 0.8  # 80% of average velocity (damped)
        self.pole_vel *= self.POLE_DRAG  # Apply drag
        self.pole_pos += self.pole_vel  # Move pole in world space
        
        # ENFORCE RIGID FORMATION: Keep all salps at fixed positions around moving pole
        # 24 iterations ensure zero drift even under thrust variations
        for _ in range(24):
            self._apply_pole_constraints()

        if self._flash_timer > 0:
            self._flash_timer -= 1

        if self._any_wall_collision():
            self._reset_all()

    # ─────────────────────────────────────────────────────────────────────
    #  Chain constraint  (position correction, spring-like)
    # ─────────────────────────────────────────────────────────────────────

    def _apply_pole_constraints(self):
        """Constrain all salps to fixed positions around central static pole.
        
        Each salp maintains a FIXED distance and angle from the pole center.
        This creates a rigid radial attachment system - like spokes on a wheel,
        but with the wheel stationary in world space.
        
        Pole does not move. Salps are pulled back to their correct spoke positions.
        """
        for i, salp in enumerate(self.salps):
            # Desired position: at pole angle + fixed distance away
            desired_pos = self.pole_pos + np.array([
                math.cos(self.pole_angles[i]) * self.pole_distances[i],
                math.sin(self.pole_angles[i]) * self.pole_distances[i],
            ])
            
            # Vector from salp to desired position
            error_vec = desired_pos - salp.pos
            error_dist = float(np.linalg.norm(error_vec))
            
            if error_dist < 1e-6:
                error_vec = np.array([1e-6, 0])
                error_dist = 1e-6
            
            direction = error_vec / error_dist
            
            # RIGID recovery: pull salp back to exact pole position with full spring force
            correction = direction * error_dist * self.POLE_SPRING_K
            salp.pos += correction
            
            # VELOCITY DAMPING: null out radial velocity (keep only angular/tangential motion)
            # This prevents salps from drifting radially away from the pole
            radial_vel = np.dot(salp.vel, direction)
            salp.vel -= direction * radial_vel * 0.35  # 35% damping of radial motion

    # ─────────────────────────────────────────────────────────────────────
    #  Collision detection
    # ─────────────────────────────────────────────────────────────────────

    def _any_wall_collision(self) -> bool:
        for s in self.salps:
            wall  = self.margin + s.max_extent
            hit_x = s.pos[0] <= wall or s.pos[0] >= self.width  - wall
            hit_y = s.pos[1] <= wall or s.pos[1] >= self.height - wall
            if hit_x or hit_y:
                return True
        return False

    def _reset_all(self):
        for s in self.salps:
            s.reset()
        self._flash_timer = self._flash_duration
    
    # ─────────────────────────────────────────────────────────────────────
    #  Autonomous control and reward calculation
    # ─────────────────────────────────────────────────────────────────────
    
    def apply_action(self, action: int, salp_idx: int = 0):
        """Apply discrete action to a salp (9 possible actions).
        
        INSTANT THRUST MODE: All salps move immediately at full power when command given.
        Water is set to full (1.0) to enable immediate thrust application.
        Chain constraints ensure synchronized movement across all salps.
        """
        s = self.salps[salp_idx]
        
        # Clear inputs
        s.nozzle_input = 0.0
        
        # Map discrete action to control
        # Action 0 = rest (no thrust), actions 1-8 = movement with full water
        if action == 0:
            # Rest: zero out water for all salps (coordinated stop)
            for salp in self.salps:
                salp.water = 0.0
        elif action == 1:
            # Steer left - full thrust, all salps
            for salp in self.salps:
                salp.water = 1.0
                salp.nozzle_input = 1.0  # ALL salps steer left
        elif action == 2:
            # Steer straight - full thrust, all salps
            for salp in self.salps:
                salp.water = 1.0
                salp.nozzle_input = 0.0  # ALL salps steer straight
        elif action == 3:
            # Steer right - full thrust, all salps
            for salp in self.salps:
                salp.water = 1.0
                salp.nozzle_input = -1.0  # ALL salps steer right
        elif action == 4:
            # Steer left (with thrust) - full thrust, all salps
            for salp in self.salps:
                salp.water = 1.0
                salp.nozzle_input = 1.0  # ALL salps steer left
        elif action == 5:
            # Steer straight (with thrust) - full thrust, all salps
            for salp in self.salps:
                salp.water = 1.0
                salp.nozzle_input = 0.0  # ALL salps steer straight
        elif action == 6:
            # Steer right (with thrust) - full thrust, all salps
            for salp in self.salps:
                salp.water = 1.0
                salp.nozzle_input = -1.0  # ALL salps steer right
        elif action == 7:
            # Steer left (no new inhale) - full thrust, all salps
            for salp in self.salps:
                salp.water = 1.0
                salp.nozzle_input = 1.0  # ALL salps steer left
        elif action == 8:
            # Steer right (no new inhale) - full thrust, all salps
            for salp in self.salps:
                salp.water = 1.0
                salp.nozzle_input = -1.0  # ALL salps steer right
    
    def calculate_reward(self, salps, goal_pos):
        """Calculate reward based on MINIMUM distance (closest salp to goal).
        
        This encourages all salps to work together to get ANY salp to the goal.
        Reward structure:
        - Base: -1 per step (encourages speed)
        - Distance: penalty based on closest salp distance
        - Goal: +100 when ANY salp reaches goal
        - Collision: -50 for hitting walls
        """
        distances = [float(np.linalg.norm(s.pos - goal_pos)) for s in salps]
        min_distance = min(distances)
        closest_salp_idx = distances.index(min_distance)
        
        # Base reward: -1 per step (encourages speed)
        reward = -1.0
        
        # Distance reward: penalty for closest salp being far from goal
        # Encourages getting ANY salp to goal efficiently
        reward += -0.01 * (min_distance / 500.0)
        
        # Goal reached: large bonus when ANY salp reaches goal
        goal_threshold = 60.0
        goal_reached = min_distance < goal_threshold
        if goal_reached:
            reward += 100.0
        
        # Wall collision penalty
        if self._any_wall_collision():
            reward -= 50.0
        
        return reward, min_distance, goal_reached, closest_salp_idx
    
    def get_closest_salp_info(self, salps, goal_pos):
        """Get info about the closest salp to goal.
        
        Returns: (distance, salp_index, salp_object)
        Useful for tracking which salp is leading toward goal.
        """
        distances = [float(np.linalg.norm(s.pos - goal_pos)) for s in salps]
        min_distance = min(distances)
        closest_idx = distances.index(min_distance)
        return min_distance, closest_idx, salps[closest_idx]

    # ─────────────────────────────────────────────────────────────────────
    #  Nozzle steering
    # ─────────────────────────────────────────────────────────────────────

    def _update_nozzle(self, s: Salp):
        s.target_nozzle = np.clip(
            s.target_nozzle + s.nozzle_input * s.nozzle_move_speed,
            -s.max_nozzle_angle, s.max_nozzle_angle,
        )
        diff = s.target_nozzle - s.nozzle_angle
        step = s.nozzle_move_speed
        s.nozzle_angle += np.sign(diff) * step if abs(diff) > step else diff

    # ─────────────────────────────────────────────────────────────────────
    #  Breathing cycle
    # ─────────────────────────────────────────────────────────────────────

    def _update_breathing(self, s: Salp):
        """INSTANT THRUST MODE: Apply thrust immediately without breathing phase.
        
        Water level is set by apply_action() to 0.0 (rest) or 1.0 (thrust).
        This method applies thrust instantly when water > 0, enabling synchronized
        movement across all salps via chain constraints.
        """
        # Maintain resting shape when no water (no thrust)
        if s.water <= 0.0:
            s.phase = "rest"
            s.semi_a = s.radius * 1.3
            s.semi_b = s.radius * 0.8
        else:
            # Actively thrusting: maintain expanded shape and apply thrust
            s.phase = "exhaling"  # Visual feedback
            # Slight body shape pulse during thrust
            s.semi_a = s.radius * (1.1 + 0.15 * (s.water * 0.5))
            s.semi_b = s.radius * (1.1 - 0.2 * (s.water * 0.5))
            # Apply thrust immediately every frame while water > 0
            self._apply_thrust(s)

    # ─────────────────────────────────────────────────────────────────────
    #  Jet thrust
    # ─────────────────────────────────────────────────────────────────────

    def _apply_thrust(self, s: Salp):
        magnitude    = s.max_thrust * s.water * 0.4
        thrust_angle = s.angle - s.nozzle_angle

        s.vel[0] += math.cos(thrust_angle) * magnitude * 0.012
        s.vel[1] += math.sin(thrust_angle) * magnitude * 0.012

        torque = (
            -s.nozzle_angle * magnitude * 0.0002
            + math.sin(-s.nozzle_angle) * magnitude * s.max_extent * 0.7 * 0.00005
            + -s.nozzle_angle * magnitude * s.water * 0.00003
        )
        s.angular_vel += torque

    # ─────────────────────────────────────────────────────────────────────
    #  Physics
    # ─────────────────────────────────────────────────────────────────────

    def _update_physics(self, s: Salp):
        s.vel         *= s.drag
        s.angular_vel *= 0.95
        s.pos         += s.vel
        s.angle        = (s.angle + s.angular_vel + math.pi) % (2 * math.pi) - math.pi

        wall = self.margin + s.max_extent
        for axis, limit in ((0, self.width), (1, self.height)):
            if s.pos[axis] < wall:
                s.pos[axis]    = wall
                s.vel[axis]    = abs(s.vel[axis]) * 0.4
                s.angular_vel *= 0.7
            elif s.pos[axis] > limit - wall:
                s.pos[axis]    = limit - wall
                s.vel[axis]    = -abs(s.vel[axis]) * 0.4
                s.angular_vel *= 0.7

    # ─────────────────────────────────────────────────────────────────────
    #  Rendering
    # ─────────────────────────────────────────────────────────────────────

    def render(self):
        # Flash red on reset, otherwise deep water
        if self._flash_timer > 0:
            intensity = int(80 * self._flash_timer / self._flash_duration)
            self.screen.fill((intensity, 5, 10))
        else:
            self.screen.fill((10, 25, 50))

        m = self.margin
        pygame.draw.rect(self.screen, (30, 60, 100),
                         (m, m, self.width - 2*m, self.height - 2*m), 3)

        # Draw pole and connectors BEHIND salps
        self._draw_pole()
        self._draw_chain_connectors()

        # Draw goal star
        self._draw_star(self.goal_pos, 16, (255, 210, 0))

        # Compute distances to goal for all salps
        dists_to_goal = [float(np.linalg.norm(s.pos - self.goal_pos)) for s in self.salps]
        closest_idx   = int(np.argmin(dists_to_goal))

        # Static yellow selection ring (no animation)
        s_sel = self.salps[self.selected]
        pygame.draw.circle(self.screen, (255, 255, 80),
                           (int(s_sel.pos[0]), int(s_sel.pos[1])),
                           int(s_sel.max_extent) + 10, 3)

        # Bright cyan/magenta square around CLOSEST salp (pulsing indicator)
        # This shows which salp is leading toward the goal
        s_close = self.salps[closest_idx]
        sq = int(s_close.max_extent) + 12
        pulse = int(5 + 3 * math.sin(self._flash_timer * 0.3))
        pygame.draw.rect(self.screen, (0, 255, 255),  # Cyan for closest
                         (int(s_close.pos[0]) - sq, int(s_close.pos[1]) - sq,
                          sq * 2, sq * 2), pulse)
        
        # Add "LEADING" text above closest salp
        closest_label = self.small.render("LEADING", True, (0, 255, 255))
        self.screen.blit(closest_label, 
                         (int(s_close.pos[0]) - closest_label.get_width() // 2,
                          int(s_close.pos[1]) - int(s_close.max_extent) - 35))

        for i, salp in enumerate(self.salps):
            self._draw_salp(salp, colour=self.COLOURS[i % len(self.COLOURS)])

        # Distance labels drawn on top of salps
        for i, s in enumerate(self.salps):
            d   = dists_to_goal[i]
            lbl = self.small.render(f"{d:.0f}px", True, (255, 230, 100))
            self.screen.blit(lbl, (int(s.pos[0]) - lbl.get_width() // 2,
                                   int(s.pos[1]) - int(s.max_extent) - 18))

        self._draw_hud()
        pygame.display.flip()
        self.clock.tick(self.fps)

    def _draw_star(self, pos, size, colour):
        """Draw a simple 5-pointed star at pos."""
        cx, cy  = float(pos[0]), float(pos[1])
        points  = []
        for k in range(10):
            angle = math.pi / 2 + k * math.pi / 5   # 36° steps
            r     = size if k % 2 == 0 else size * 0.45
            points.append((cx + math.cos(angle) * r,
                           cy - math.sin(angle) * r))
        pygame.draw.polygon(self.screen, colour, points)

    def _draw_pole(self):
        """Draw static central pole and spokes to each salp."""
        pole_screen = (int(self.pole_pos[0]), int(self.pole_pos[1]))
        
        # Draw pole as vertical line (thick cylinder)
        pole_radius = 8
        pygame.draw.circle(self.screen, (180, 160, 140),
                          pole_screen, pole_radius)
        pygame.draw.circle(self.screen, (220, 200, 170),
                          pole_screen, pole_radius, 2)
        
        # Draw spokes from pole to each salp (connecting ropes)
        spoke_colour = (120, 140, 160)
        for i, salp in enumerate(self.salps):
            salp_screen = (int(salp.pos[0]), int(salp.pos[1]))
            pygame.draw.line(self.screen, spoke_colour,
                            pole_screen, salp_screen, 3)
            
            # Small joint marker at spoke attachment to salp
            pygame.draw.circle(self.screen, (200, 220, 240),
                              salp_screen, 5, 1)

    def _draw_chain_connectors(self):
        """Draw stick + joint dots between each adjacent salp pair."""
        for i in range(len(self.salps) - 1):
            a = self.salps[i]
            b = self.salps[i + 1]

            a_pos = (int(a.pos[0]), int(a.pos[1]))
            b_pos = (int(b.pos[0]), int(b.pos[1]))

            # Direction from a to b
            delta = b.pos - a.pos
            dist  = float(np.linalg.norm(delta))
            if dist < 1e-6:
                continue
            unit = delta / dist

            # Stick endpoints: start at surface of a, end at surface of b
            stick_start = a.pos + unit * a.max_extent * 0.95
            stick_end   = b.pos - unit * b.max_extent * 0.95

            sx, sy = int(stick_start[0]), int(stick_start[1])
            ex, ey = int(stick_end[0]),   int(stick_end[1])

            pygame.draw.line(self.screen, self.STICK_COLOUR,
                             (sx, sy), (ex, ey), self.STICK_WIDTH)

            # Small joint dots at each end of the stick
            pygame.draw.circle(self.screen, self.JOINT_COLOUR, (sx, sy), self.JOINT_RADIUS)
            pygame.draw.circle(self.screen, self.JOINT_COLOUR, (ex, ey), self.JOINT_RADIUS)

    def _draw_salp(self, s: Salp, colour: tuple):
        rx, ry = int(s.pos[0]), int(s.pos[1])

        tint_delta = {
            "rest":     (  0,   0,   0),
            "inhaling": (-30, -40, -30),
            "exhaling": ( 50, -40, -80),
        }.get(s.phase, (0, 0, 0))
        body_col = tuple(max(0, min(255, c + d)) for c, d in zip(colour, tint_delta))

        ew, eh = int(s.semi_a * 2), int(s.semi_b * 2)
        if ew > 0 and eh > 0:
            surf = pygame.Surface((ew, eh), pygame.SRCALPHA)
            pygame.draw.ellipse(surf, body_col, (0, 0, ew, eh))
            rot  = pygame.transform.rotate(surf, -math.degrees(s.angle))
            self.screen.blit(rot, rot.get_rect(center=(rx, ry)))

        pygame.draw.circle(self.screen,
                           tuple(max(0, c - 50) for c in colour),
                           (rx, ry), int(s.max_extent), 2)

        fx = rx + math.cos(s.angle) * s.max_extent * 0.8
        fy = ry + math.sin(s.angle) * s.max_extent * 0.8
        pygame.draw.circle(self.screen, (255, 255, 255), (int(fx), int(fy)), 4)

        nozzle_root_x = rx + math.cos(s.angle + math.pi) * s.max_extent * 0.9
        nozzle_root_y = ry + math.sin(s.angle + math.pi) * s.max_extent * 0.9
        nozzle_dir    = s.angle + math.pi + s.nozzle_angle
        nozzle_len    = 15
        nozzle_tip_x  = nozzle_root_x + math.cos(nozzle_dir) * nozzle_len
        nozzle_tip_y  = nozzle_root_y + math.sin(nozzle_dir) * nozzle_len
        pygame.draw.line(self.screen, (220, 210, 90),
                         (int(nozzle_root_x), int(nozzle_root_y)),
                         (int(nozzle_tip_x),  int(nozzle_tip_y)), 4)

        if s.phase == "exhaling":
            perp = nozzle_dir + math.pi / 2
            for j in range(8):
                dist  = nozzle_len + 5 + j * 4
                curve = abs(s.nozzle_angle) * 0.5 * j * 0.3 * (-1 if s.nozzle_angle > 0 else 1)
                px    = nozzle_root_x + math.cos(nozzle_dir) * dist + math.cos(perp) * curve
                py    = nozzle_root_y + math.sin(nozzle_dir) * dist + math.sin(perp) * curve
                pygame.draw.circle(self.screen, (80, 120, max(100, 200 - j * 15)),
                                   (int(px), int(py)), max(1, 5 - j))

        if s.speed > 0.5:
            pygame.draw.line(self.screen, (0, 220, 120),
                             (rx, ry),
                             (int(rx + s.vel[0] * 8), int(ry + s.vel[1] * 8)), 2)

    def _draw_hud(self):
        self.screen.blit(
            self.font.render(
                "SALP Chain (DQN Multi-Agent)  |  SPACE: Toggle Auto/Manual  |  W=inhale  A/D=steer  |  ESC: quit",
                True, (200, 200, 200)
            ), (10, 10)
        )

        mode_text = "AUTO (Learning)" if self.autonomous_mode else "MANUAL Control"
        self.screen.blit(
            self.font.render(f"Mode: {mode_text}", True, (100, 255, 100) if self.autonomous_mode else (255, 200, 100)),
            (10, 35)
        )
        
        # Calculate closest salp info for display
        dists_to_goal = [float(np.linalg.norm(s.pos - self.goal_pos)) for s in self.salps]
        min_dist = min(dists_to_goal)
        closest_idx = dists_to_goal.index(min_dist)
        
        self.screen.blit(
            self.small.render(f"Episode Steps: {self.episode_steps}  Total Reward: {self.total_reward:.1f}  "
                            f"Closest: Salp {closest_idx} ({min_dist:.0f}px)", True, (200, 200, 200)),
            (10, 60)
        )

        for i, s in enumerate(self.salps):
            col  = self.COLOURS[i % len(self.COLOURS)]
            is_closest = "★ CLOSEST" if i == closest_idx else ""
            is_selected = "◀ SELECTED" if i == self.selected else ""
            tag = f" {is_closest} {is_selected}".strip()
            
            line = (f"Salp {i}  |  {s.phase.upper():<9}"
                    f"  water={s.water:.2f}  spd={s.speed:.1f}"
                    f"  noz={math.degrees(s.nozzle_angle):+.0f}°")
            if tag:
                line += f"  {tag}"
            
            self.screen.blit(
                self.small.render(line, True, col),
                (10, 80 + i * 16)
            )

    def close(self):
        pygame.quit()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    STICK_LEN = SalpEnv.STICK_LEN

    # Salps are now smaller (half the previous size)
    radii       = [13, 15, 17, 15, 13]
    max_thrusts = [90, 90, 100, 90, 90]

    xs = [0.0]
    for i in range(1, 5):
        gap = radii[i-1] * 1.3 + radii[i] * 1.3 + STICK_LEN
        xs.append(xs[-1] + gap)

    chain_width = xs[-1] - xs[0]

    # Random spawn: pick a random centre x,y within the tank interior
    margin  = 50
    padding = chain_width / 2 + 60
    cx = random.uniform(margin + padding, 1100 - margin - padding)
    cy = random.uniform(margin + 60,      620  - margin - 60)
    offset_x = cx - (xs[0] + chain_width / 2)
    xs = [x + offset_x for x in xs]

    salps = [
        Salp(radius=radii[i], start_pos=(xs[i], cy), max_thrust=max_thrusts[i])
        for i in range(5)
    ]

    # Create environment with INSTANT decision-making for responsive control
    # width=1600, height=900 (bigger than 1100x620)
    # margin=60 (scaled for bigger environment)
    # thinking_frames=1 means agent decides every frame (fully responsive 60 decisions/sec at 60 FPS)
    env = SalpEnv(salps=salps, width=1600, height=900, margin=60, fps=60, thinking_frames=1)
    
    # Initialize DQN agent (controls salp 0)
    # State size: 22 (2 pos + 2 vel + 1 angle + 2 goal + 5 distances + 10 all-salp velocities)
    # Agent observes all salp velocities for better movement prediction
    agent = DQNAgent(state_size=22, action_size=9)
    agent.train_mode()
    
    # Training parameters
    episodes = 100
    target_update_interval = 5
    update_target_counter = 0
    controlled_salp = 0

    # Number keys 1-5 map to salp indices 0-4
    num_keys = [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5]

    running = True
    episode = 0
    
    while running and episode < episodes:
        # Reset environment for new episode
        for s in env.salps:
            s.reset()
        inner = env.margin + 60
        env.goal_pos = np.array([
            random.uniform(inner, env.width - inner),
            random.uniform(inner, env.height - inner),
        ])
        
        env.episode_steps = 0
        env.total_reward = 0.0
        state = agent.get_state(env.salps, env.goal_pos, env.width, env.height)
        
        # For thinking delay: track current decision and frame counter
        current_action = 0
        env.thinking_counter = 0
        
        episode_done = False
        while not episode_done and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        # Toggle autonomous/manual mode
                        env.autonomous_mode = not env.autonomous_mode
                    # Keys 1-5 select a salp for manual control
                    for idx, k in enumerate(num_keys):
                        if event.key == k:
                            env.selected = idx
                            controlled_salp = idx

            keys = pygame.key.get_pressed()

            # Clear all inputs
            for s in env.salps:
                s.inhale_held = False
                s.nozzle_input = 0.0

            if env.autonomous_mode:
                # Agent thinks periodically (every thinking_frames frames)
                if env.thinking_counter >= env.thinking_frames:
                    # Time to make a new decision
                    action = agent.select_action(state, training=True)
                    current_action = action
                    env.thinking_counter = 0  # Reset thinking counter
                else:
                    # Still thinking: repeat last action
                    action = current_action
                    env.thinking_counter += 1
                
                env.apply_action(action, controlled_salp)
            else:
                # Manual control: convert keys to instant thrust
                # With all-salp direction control, all salps steer together
                if keys[pygame.K_w]:
                    # Thrust engaged: set all salps to full water
                    for s in env.salps:
                        s.water = 1.0
                    # Steering applied to ALL salps (not just selected)
                    nozzle_direction = (
                        1.0 if keys[pygame.K_a] else
                        -1.0 if keys[pygame.K_d] else
                        0.0
                    )
                    for s in env.salps:
                        s.nozzle_input = nozzle_direction
                else:
                    # No thrust: rest mode
                    for s in env.salps:
                        s.water = 0.0
                        s.nozzle_input = 0.0

            # Step environment
            env.step()
            
            # Get reward and next state (based on all salps' positions)
            reward, min_dist_to_goal, goal_reached, closest_salp_idx = env.calculate_reward(env.salps, env.goal_pos)
            next_state = agent.get_state(env.salps, env.goal_pos, env.width, env.height)
            
            # Check if episode is done
            # Episode ends if:
            # 1. ANY salp reaches the goal
            # 2. Hit wall/collision
            # 3. Max steps exceeded
            done = (goal_reached or 
                    env._any_wall_collision() or 
                    env.episode_steps >= env.max_episode_steps)
            
            if env.autonomous_mode:
                # Store experience in memory
                agent.remember(state, action, reward, next_state, done)
                
                # Train on batch
                if len(agent.memory) >= agent.batch_size:
                    agent.replay()
            
            env.episode_steps += 1
            env.total_reward += reward
            state = next_state
            
            # Render
            env.render()
            
            if done:
                episode_done = True

        # Update target network periodically
        if env.autonomous_mode:
            update_target_counter += 1
            if update_target_counter >= target_update_interval:
                agent.update_target_network()
                update_target_counter = 0
                
                # Decay epsilon
                agent.decay_epsilon()
                
                # Determine result message
                if goal_reached:
                    result = f"✓ GOAL! (Salp {closest_salp_idx})"
                elif env._any_wall_collision():
                    result = "✗ Collision"
                else:
                    result = "⊘ Time limit"
                
                print(f"Episode {episode + 1}/{episodes} | {result:20} | "
                      f"Reward: {env.total_reward:+7.2f} | Dist: {min_dist_to_goal:6.1f}px | ε: {agent.epsilon:.3f}")
        
        episode += 1

    env.close()


if __name__ == "__main__":
    main()
