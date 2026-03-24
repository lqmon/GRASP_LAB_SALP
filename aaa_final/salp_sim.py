"""
SALP Robot Simulation  –  Flexible Spring Chain + Deep Q Learning
==================================================================
Five salps connected 1-2-3-4-5 by spring constraints (no rigid pole).
Each salp has its own position, velocity, and thruster.
The chain can flex, curve, and snake — it is NOT rigid.

Physics
-------
  • Each salp: pos, vel, drag, wall bounce
  • Adjacent salps linked by a spring (rest length = LINK_LEN).
    Spring is soft enough to allow natural flexing but stiff enough
    to hold the chain together.
  • Each salp fires its own thruster independently; direction is
    decided cooperatively (own goal-angle blended with chain consensus).

DQN Agent
---------
  • Controls ALL 5 salps simultaneously: one action = one thrust+steer
    direction applied to every salp.
  • Trains MULTIPLE gradient steps per environment step (THINK_STEPS).
  • State includes every salp's pos, vel, angle-to-goal, link stretch.
  • Double DQN + Dueling + Prioritised Replay.

Controls (manual mode)
---------
  W        : thrust all salps forward
  A / D    : steer left / right
  Space    : toggle auto / manual
  R        : reset episode
  ESC      : quit
"""

import math, random
import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

N_SALPS      = 5
LINK_LEN     = 68        # spring rest length (px)
SPRING_K     = 0.35      # spring stiffness per frame
DRAG         = 0.955     # per-salp drag
MAX_SPEED    = 10.0
THRUST_MAG   = 0.60
GOAL_RADIUS  = 30

THINK_STEPS  = 4         # gradient steps per env step


# ─────────────────────────────────────────────────────────────────────────────
#  DUELING DQN
# ─────────────────────────────────────────────────────────────────────────────

class DuelingDQN(nn.Module):
    def __init__(self, state_sz, action_sz, hidden=512):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_sz, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, 256),      nn.ReLU(),
        )
        self.val = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        self.adv = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, action_sz))

    def forward(self, x):
        h = self.body(x)
        v = self.val(h)
        a = self.adv(h)
        return v + a - a.mean(-1, keepdim=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PRIORITISED REPLAY
# ─────────────────────────────────────────────────────────────────────────────

class PER:
    def __init__(self, cap=300000, alpha=0.6):
        self.cap, self.alpha = cap, alpha
        self.buf, self.pri   = [], []
        self.pos             = 0

    def push(self, *t):
        p = max(self.pri) if self.pri else 1.0
        if len(self.buf) < self.cap:
            self.buf.append(t); self.pri.append(p)
        else:
            self.buf[self.pos] = t; self.pri[self.pos] = p
        self.pos = (self.pos + 1) % self.cap

    def sample(self, n, beta=0.5):
        p  = np.array(self.pri) ** self.alpha
        p /= p.sum()
        idx = np.random.choice(len(self.buf), n, p=p, replace=False)
        w   = (len(self.buf) * p[idx]) ** (-beta)
        w  /= w.max()
        return [self.buf[i] for i in idx], idx, w.astype(np.float32)

    def update(self, idx, errs):
        for i, e in zip(idx, errs):
            self.pri[i] = float(abs(e)) + 1e-6

    def __len__(self): return len(self.buf)


# ─────────────────────────────────────────────────────────────────────────────
#  AGENT
# ─────────────────────────────────────────────────────────────────────────────

class Agent:
    """
    State (37 dims):
      Per salp (5 x 6 = 30):
        norm_x, norm_y, norm_vx, norm_vy,
        sin(angle_to_goal), cos(angle_to_goal)
      Chain-level (7):
        centroid_x, centroid_y,
        centroid_vx, centroid_vy,
        consensus_sin, consensus_cos,
        mean_link_stretch

    Actions (9):
      0=coast, 1=N, 2=NE, 3=E, 4=SE, 5=S, 6=SW, 7=W, 8=NW
    """
    STATE_SIZE  = N_SALPS * 6 + 7   # 37
    ACTION_SIZE = 9

    _DIRS = [
        (0.0, 0.0),
        (0.0,-1.0),(1.0,-1.0),(1.0, 0.0),(1.0, 1.0),
        (0.0, 1.0),(-1.0,1.0),(-1.0,0.0),(-1.0,-1.0),
    ]

    def __init__(self, lr=3e-4, gamma=0.99,
                 eps=0.1, eps_decay=0.9985, eps_min=0.05,
                 batch=128, target_update=300):
        self.gamma       = gamma
        self.eps         = eps
        self.eps_decay   = eps_decay
        self.eps_min     = eps_min
        self.batch       = batch
        self.tgt_update  = target_update
        self.steps       = 0
        self.beta        = 0.4

        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q   = DuelingDQN(self.STATE_SIZE, self.ACTION_SIZE).to(self.dev)
        self.tq  = DuelingDQN(self.STATE_SIZE, self.ACTION_SIZE).to(self.dev)
        self.tq.load_state_dict(self.q.state_dict())
        self.tq.eval()
        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.mem = PER(cap=300000)

    def state_of(self, salps, goal, W, H):
        diag = math.hypot(W, H)
        feats = []
        sin_s = cos_s = 0.0
        cvx = sum(s.vel[0] for s in salps) / N_SALPS
        cvy = sum(s.vel[1] for s in salps) / N_SALPS
        cx  = sum(s.pos[0] for s in salps) / N_SALPS
        cy  = sum(s.pos[1] for s in salps) / N_SALPS

        for s in salps:
            dx = goal[0] - s.pos[0]
            dy = goal[1] - s.pos[1]
            ang = math.atan2(dy, dx)
            sin_s += math.sin(ang)
            cos_s += math.cos(ang)
            feats += [
                s.pos[0] / W,
                s.pos[1] / H,
                float(np.clip(s.vel[0] / MAX_SPEED, -1, 1)),
                float(np.clip(s.vel[1] / MAX_SPEED, -1, 1)),
                math.sin(ang),
                math.cos(ang),
            ]

        stretches = []
        for i in range(N_SALPS - 1):
            d = float(np.linalg.norm(salps[i+1].pos - salps[i].pos))
            stretches.append((d - LINK_LEN) / LINK_LEN)
        mean_stretch = sum(stretches) / len(stretches)

        feats += [
            cx / W, cy / H,
            float(np.clip(cvx / MAX_SPEED, -1, 1)),
            float(np.clip(cvy / MAX_SPEED, -1, 1)),
            sin_s / N_SALPS, cos_s / N_SALPS,
            float(np.clip(mean_stretch, -1, 1)),
        ]
        return np.array(feats, dtype=np.float32)

    def thrust_vec(self, action):
        dx, dy = self._DIRS[action]
        if dx == 0.0 and dy == 0.0:
            return np.array([0.0, 0.0])
        l = math.hypot(dx, dy)
        return np.array([dx / l, dy / l]) * THRUST_MAG

    def act(self, state, train=True):
        if train and random.random() < self.eps:
            return random.randrange(self.ACTION_SIZE)
        t = torch.FloatTensor(state).unsqueeze(0).to(self.dev)
        with torch.no_grad():
            return self.q(t).argmax(1).item()

    def push(self, s, a, r, ns, d):
        self.mem.push(s, a, r, ns, d)

    def learn(self):
        if len(self.mem) < self.batch:
            return None
        losses = []
        self.beta = min(1.0, self.beta + 2e-5)

        for _ in range(THINK_STEPS):
            batch, idx, w = self.mem.sample(self.batch, self.beta)
            s, a, r, ns, d = zip(*batch)
            S  = torch.FloatTensor(np.array(s)).to(self.dev)
            A  = torch.LongTensor(a).to(self.dev)
            R  = torch.FloatTensor(r).to(self.dev)
            NS = torch.FloatTensor(np.array(ns)).to(self.dev)
            D  = torch.FloatTensor(d).to(self.dev)
            W  = torch.FloatTensor(w).to(self.dev)

            q_val = self.q(S).gather(1, A.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                na  = self.q(NS).argmax(1, keepdim=True)
                nq  = self.tq(NS).gather(1, na).squeeze(1)
                tgt = R + (1 - D) * self.gamma * nq

            td   = (q_val - tgt).detach().cpu().numpy()
            loss = (W * F.smooth_l1_loss(q_val, tgt, reduction='none')).mean()
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
            self.opt.step()
            self.mem.update(idx, td)
            losses.append(loss.item())

            self.steps += 1
            
            if self.steps % self.tgt_update == 0:
                self.tq.load_state_dict(self.q.state_dict())

        return sum(losses) / len(losses)

    def decay(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)


# ─────────────────────────────────────────────────────────────────────────────
#  SALP PARTICLE
# ─────────────────────────────────────────────────────────────────────────────

class Salp:
    def __init__(self, radius, pos):
        self.radius    = radius
        self.semi_a    = radius * 1.3
        self.semi_b    = radius * 0.8
        self.pos       = np.array(pos, dtype=float)
        self.vel       = np.zeros(2)
        self.nozzle    = 0.0
        self.phase     = "rest"
        self.thrust_on = False

    @property
    def max_extent(self): return max(self.semi_a, self.semi_b)

    def reset(self, pos):
        self.pos       = np.array(pos, dtype=float)
        self.vel       = np.zeros(2)
        self.nozzle    = 0.0
        self.phase     = "rest"
        self.thrust_on = False


# ─────────────────────────────────────────────────────────────────────────────
#  ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────

class Env:
    COLOURS = [
        (100, 165, 230),
        ( 80, 205, 165),
        (210, 160, 100),
        (195, 115, 175),
        (165, 115, 215),
    ]

    def __init__(self, W=1400, H=800, margin=70, fps=60):
        self.W, self.H   = W, H
        self.margin      = margin
        self.fps         = fps
        self.autonomous  = True

        radii = [14, 16, 18, 16, 14]
        cx = W / 2 - (N_SALPS - 1) * LINK_LEN / 2
        self.salps = [Salp(radii[i], (cx + i * LINK_LEN, H / 2)) for i in range(N_SALPS)]

        self.goal_pos      = self._rand_goal()
        self.episode_steps = 0
        self.max_steps     = 500
        self.total_reward  = 0.0
        self._prev_dist    = None
        self._flash        = 0
        self._flash_dur    = 22

        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("SALP Chain  -  Deep Q Learning")
        self.clock  = pygame.time.Clock()
        self.font   = pygame.font.Font(None, 22)
        self.small  = pygame.font.Font(None, 17)

    def _rand_goal(self):
        inn = self.margin + 90
        return np.array([random.uniform(inn, self.W - inn),
                         random.uniform(inn, self.H - inn)])

    def _centroid(self):
        return np.mean([s.pos for s in self.salps], axis=0)

    # ── spring constraints ────────────────────────────────────────────────

    def _apply_springs(self):
        for _ in range(8):
            for i in range(N_SALPS - 1):
                a, b  = self.salps[i], self.salps[i + 1]
                delta = b.pos - a.pos
                dist  = float(np.linalg.norm(delta))
                if dist < 1e-6:
                    continue
                unit  = delta / dist
                err   = dist - LINK_LEN
                corr  = unit * err * SPRING_K * 0.5
                a.pos += corr
                b.pos -= corr
                # velocity damping along link axis
                rel_v = b.vel - a.vel
                damp  = np.dot(rel_v, unit) * unit * 0.06
                a.vel += damp
                b.vel -= damp

    # ── cooperative nozzle directions ─────────────────────────────────────

    def _update_nozzles(self):
        angles = []
        for s in self.salps:
            v = self.goal_pos - s.pos
            angles.append(math.atan2(v[1], v[0]))

        sin_c = sum(math.sin(a) for a in angles) / N_SALPS
        cos_c = sum(math.cos(a) for a in angles) / N_SALPS
        consensus = math.atan2(sin_c, cos_c)

        for i, s in enumerate(self.salps):
            own  = angles[i]
            bs   = 0.45 * math.sin(own) + 0.55 * math.sin(consensus)
            bc   = 0.45 * math.cos(own) + 0.55 * math.cos(consensus)
            target = math.atan2(bs, bc)
            diff = target - s.nozzle
            diff = (diff + math.pi) % (2 * math.pi) - math.pi
            s.nozzle += float(np.clip(diff * 0.14, -0.07, 0.07))

    # ── physics step ──────────────────────────────────────────────────────

    def step(self, thrust_vec: np.ndarray):
        thrusting = float(np.linalg.norm(thrust_vec)) > 1e-6
        for s in self.salps:
            s.thrust_on = thrusting
            s.phase     = "exhaling" if thrusting else "rest"
            s.vel      += thrust_vec
            s.vel      *= DRAG
            spd = float(np.linalg.norm(s.vel))
            if spd > MAX_SPEED:
                s.vel *= MAX_SPEED / spd
            s.pos += s.vel

        self._apply_springs()
        self._wall_bounce_all()
        self._update_nozzles()
        if self._flash > 0:
            self._flash -= 1
        self.episode_steps += 1
        self.total_reward-=10

    def _wall_bounce_all(self):
        for s in self.salps:
            r    = s.max_extent
            lo_x = self.margin + r;  hi_x = self.W - self.margin - r
            lo_y = self.margin + r;  hi_y = self.H - self.margin - r
            if s.pos[0] < lo_x: s.pos[0] = lo_x; s.vel[0] =  abs(s.vel[0]) * 0.35
            if s.pos[0] > hi_x: s.pos[0] = hi_x; s.vel[0] = -abs(s.vel[0]) * 0.35
            if s.pos[1] < lo_y: s.pos[1] = lo_y; s.vel[1] =  abs(s.vel[1]) * 0.35
            if s.pos[1] > hi_y: s.pos[1] = hi_y; s.vel[1] = -abs(s.vel[1]) * 0.35

    def _any_wall(self):
        for s in self.salps:
            r = s.max_extent
            if (s.pos[0] < self.margin + r or s.pos[0] > self.W - self.margin - r or
                s.pos[1] < self.margin + r or s.pos[1] > self.H - self.margin - r):
                return True
        return False

    # ── reward ────────────────────────────────────────────────────────────

    def reward(self):
        cent  = self._centroid()
        dist  = float(np.linalg.norm(cent - self.goal_pos))
        min_d = min(float(np.linalg.norm(s.pos - self.goal_pos)) for s in self.salps)

        if self._prev_dist is None:
            self._prev_dist = dist

        delta = self._prev_dist - dist
        self._prev_dist = dist
        r = delta * 1.0 - 0.4

        done = False
        if min_d < GOAL_RADIUS:
            r   += 250.0
            done = True
        if self._any_wall():
            r -= 25.0
        if self.episode_steps >= self.max_steps:
            done = True

        return r, min_d, done

    # ── reset ─────────────────────────────────────────────────────────────

    def reset(self):
        heading = random.uniform(-math.pi, math.pi)
        ax, ay  = math.cos(heading), math.sin(heading)
        half    = (N_SALPS - 1) * LINK_LEN / 2
        inn     = self.margin + half + 40
        cx = random.uniform(inn, self.W - inn)
        cy = random.uniform(inn, self.H - inn)
        for i, s in enumerate(self.salps):
            offset = (i - (N_SALPS - 1) / 2) * LINK_LEN
            s.reset(np.array([cx + ax * offset, cy + ay * offset]))

        self.goal_pos      = self._rand_goal()
        self.episode_steps = 0
        self.total_reward  = 0.0
        self._prev_dist    = None
        self._flash        = 0

    # ── rendering ─────────────────────────────────────────────────────────

    def render(self, ep, total_ep, eps, loss, action):
        bg = (int(55 * self._flash / self._flash_dur), 5, 10) if self._flash else (8, 20, 45)
        self.screen.fill(bg)
        m = self.margin
        pygame.draw.rect(self.screen, (25, 55, 95),
                         (m, m, self.W - 2*m, self.H - 2*m), 3)

        self._draw_goal()
        self._draw_links()
        for i, s in enumerate(self.salps):
            self._draw_salp(s, self.COLOURS[i], i)
        self._draw_hud(ep, total_ep, eps, loss, action)
        pygame.display.flip()
        self.clock.tick(self.fps)

    def _draw_goal(self):
        gx, gy = int(self.goal_pos[0]), int(self.goal_pos[1])
        pygame.draw.circle(self.screen, (160, 135, 20), (gx, gy), GOAL_RADIUS, 1)
        pts = []
        for k in range(10):
            a = math.pi / 2 + k * math.pi / 5
            r = 18 if k % 2 == 0 else 8
            pts.append((gx + math.cos(a) * r, gy - math.sin(a) * r))
        pygame.draw.polygon(self.screen, (255, 210, 0), pts)

    def _draw_links(self):
        for i in range(N_SALPS - 1):
            a, b = self.salps[i], self.salps[i + 1]
            d    = float(np.linalg.norm(b.pos - a.pos))
            stretch = min(1.0, abs(d - LINK_LEN) / LINK_LEN)
            lc  = (int(50 + 180 * stretch), int(160 - 120 * stretch), 180)
            pygame.draw.line(self.screen, lc,
                             (int(a.pos[0]), int(a.pos[1])),
                             (int(b.pos[0]), int(b.pos[1])), 5)
            for p in (a.pos, b.pos):
                pygame.draw.circle(self.screen, (210, 225, 245),
                                   (int(p[0]), int(p[1])), 5, 1)

    def _draw_salp(self, s: Salp, colour, idx):
        rx, ry = int(s.pos[0]), int(s.pos[1])
        spd    = float(np.linalg.norm(s.vel))
        body_a = math.atan2(s.vel[1], s.vel[0]) if spd > 0.3 else s.nozzle

        ew, eh = int(s.semi_a * 2), int(s.semi_b * 2)
        if ew > 1 and eh > 1:
            surf = pygame.Surface((ew, eh), pygame.SRCALPHA)
            alpha = 225 if s.thrust_on else 170
            pygame.draw.ellipse(surf, colour + (alpha,), (0, 0, ew, eh))
            rot = pygame.transform.rotate(surf, -math.degrees(body_a))
            self.screen.blit(rot, rot.get_rect(center=(rx, ry)))

        pygame.draw.circle(self.screen,
                           tuple(max(0, c - 45) for c in colour),
                           (rx, ry), int(s.max_extent), 2)

        # Nozzle pointing toward cooperative goal direction
        noz_dir = s.nozzle + math.pi
        nr_x = rx + math.cos(s.nozzle - math.pi) * s.max_extent * 0.88
        nr_y = ry + math.sin(s.nozzle - math.pi) * s.max_extent * 0.88
        nt_x = nr_x + math.cos(noz_dir) * 14
        nt_y = nr_y + math.sin(noz_dir) * 14
        pygame.draw.line(self.screen, (235, 210, 75),
                         (int(nr_x), int(nr_y)), (int(nt_x), int(nt_y)), 4)

        if s.thrust_on:
            for j in range(7):
                px = nr_x + math.cos(noz_dir) * (16 + j * 5)
                py = nr_y + math.sin(noz_dir) * (16 + j * 5)
                pygame.draw.circle(self.screen,
                                   (70, 110, max(90, 195 - j * 14)),
                                   (int(px), int(py)), max(1, 5 - j))

        # Eye
        fx = rx + math.cos(s.nozzle) * s.max_extent * 0.75
        fy = ry + math.sin(s.nozzle) * s.max_extent * 0.75
        pygame.draw.circle(self.screen, (255, 255, 255), (int(fx), int(fy)), 4)

        # Velocity arrow
        if spd > 0.5:
            pygame.draw.line(self.screen, (0, 230, 110),
                             (rx, ry),
                             (int(rx + s.vel[0] * 10), int(ry + s.vel[1] * 10)), 2)

        lbl = self.small.render(str(idx), True, (200, 220, 255))
        self.screen.blit(lbl, (rx - 4, ry - int(s.max_extent) - 16))

    def _draw_hud(self, ep, total_ep, eps, loss, action):
        cent  = self._centroid()
        dist  = float(np.linalg.norm(cent - self.goal_pos))
        speed = float(np.mean([np.linalg.norm(s.vel) for s in self.salps]))

        self.screen.blit(self.font.render(
            "SALP Chain DQN  |  SPACE: Auto/Manual  |  W: thrust  A/D: steer  R: reset  ESC: quit",
            True, (185, 200, 215)), (10, 10))

        mc = (75, 255, 115) if self.autonomous else (255, 200, 75)
        mt = f"AUTO  (thinking x{THINK_STEPS})" if self.autonomous else "MANUAL"
        self.screen.blit(self.font.render(f"Mode: {mt}", True, mc), (10, 31))

        ls = f"{loss:.5f}" if loss else "---"
        dirs = ["coast","N","NE","E","SE","S","SW","W","NW"]
        self.screen.blit(self.small.render(
            f"Ep {ep}/{total_ep}  Step {self.episode_steps}  "
            f"Reward {self.total_reward:+.1f}  Dist {dist:.0f}px  "
            f"Avg spd {speed:.2f}  eps {eps:.3f}  Loss {ls}  Act: {dirs[action]}",
            True, (185, 185, 185)), (10, 54))

        for i, s in enumerate(self.salps):
            d   = float(np.linalg.norm(s.pos - self.goal_pos))
            spd = float(np.linalg.norm(s.vel))
            line = f"Salp {i}  dist={d:.0f}px  spd={spd:.2f}  noz={math.degrees(s.nozzle):+.0f}deg"
            self.screen.blit(self.small.render(line, True, self.COLOURS[i]),
                             (10, 72 + i * 15))

        stretch_strs = []
        for i in range(N_SALPS - 1):
            d = float(np.linalg.norm(self.salps[i+1].pos - self.salps[i].pos))
            stretch_strs.append(f"{i}-{i+1}:{d:.0f}px")
        self.screen.blit(self.small.render(
            "Links: " + "  ".join(stretch_strs), True, (130, 160, 200)),
            (10, 72 + N_SALPS * 15 + 4))

    def close(self):
        pygame.quit()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    env       = Env(W=1400, H=800, margin=70, fps=60)
    agent     = Agent()
    total_ep  = 1000
    ep        = 0
    loss_disp = None
    action    = 0

    env.reset()
    state = agent.state_of(env.salps, env.goal_pos, env.W, env.H)

    running = True
    while running and ep < total_ep:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    env.autonomous = not env.autonomous
                elif event.key == pygame.K_r:
                    env.reset()
                    state = agent.state_of(env.salps, env.goal_pos, env.W, env.H)

        keys = pygame.key.get_pressed()
        if env.autonomous:
            action = agent.act(state, train=True)
        else:
            w = keys[pygame.K_w]
            a = keys[pygame.K_a]
            d = keys[pygame.K_d]
            if   w and a: action = 8
            elif w and d: action = 2
            elif w:       action = 1
            elif a:       action = 7
            elif d:       action = 3
            else:         action = 0

        tv = agent.thrust_vec(action)

        env.step(tv)
        r, dist, done = env.reward()
        env.total_reward += r
        next_state = agent.state_of(env.salps, env.goal_pos, env.W, env.H)

        if env.autonomous:
            agent.push(state, action, r, next_state, float(done))
            loss = agent.learn()
            if loss is not None:
                loss_disp = loss

        state = next_state
        env.render(ep + 1, total_ep, agent.eps, loss_disp, action)

        if done:
            if env.autonomous:
                agent.decay()
            goal_hit = dist < GOAL_RADIUS
            result   = "GOAL" if goal_hit else "timeout"
            print(f"Ep {ep+1:4d}/{total_ep}  {result:<9}  "
                  f"Reward {env.total_reward:+8.2f}  Dist {dist:6.1f}px  "
                  f"Steps {env.episode_steps:4d}  eps {agent.eps:.4f}  "
                  f"Mem {len(agent.mem):5d}")
            ep += 1
            env.reset()
            state = agent.state_of(env.salps, env.goal_pos, env.W, env.H)

    env.close()


if __name__ == "__main__":
    main()