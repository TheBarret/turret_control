import pygame
import random
import numpy as np

from typing import List
from dataclasses import dataclass

@dataclass
class Config:
    # sim settings
    FPS = 60
    WINDOW_SIZE = (1024, 600)
    
    # physics settings
    GRAVITY = 9.81 / FPS
    AIR_RESISTANCE = 0.01
    
    # turret settings
    BARREL_LENGTH = 20
    MIN_ANGLE = 0
    MAX_ANGLE = 90
    ANGLE_STEP = 1
    MIN_POWER = 5
    MAX_POWER = 50
    POWER_STEP = 1
    
    COLORS = {
        'BLACK': (0, 0, 0),
        'WHITE': (255, 255, 255),
        'RED': (255, 0, 0),
        'GREEN': (0, 255, 0),
        'BLUE': (0, 0, 255),
        'GRAY': (128, 128, 128),
    }

class Projectile:
    def __init__(self, pos: np.ndarray, angle: float, power: float, config: Config):
        self.config = config
        self.pos = pos.copy()
        self.origin = pos.copy()
        self.vel = np.array([np.cos(np.radians(angle)) * power, -np.sin(np.radians(angle)) * power])
        self.alive = True
        self.lifetime = 0
        self.lifetime_m = config.FPS * 10
        self.trajectory = [pos.copy()]
        self.contact = None
        self.flight_time = 0
        self.radius = 4
        self.particles = []
        self.hitdata = ''
        self.trail_config = {
            'size_range': (1, 5),           # (min, max) particle sizes
            'lifetime_range': (5, 20),      # (min, max) particle lifetime in frames
            'emission_rate': 2,             # particles per frame
            'speed_range': (0, 0.8),        # (min, max) particle speed
            'spread_angle': 25,             # degrees of spread from center
            'color_map': {5: (128, 128, 128), 10: (128, 128, 128), 20: (128, 128, 128)}
        }
        self.explosion_config = {
            'particle_count': 10,
            'size_range': (20, 20),
            'speed_range': (1, 3.0),
            'lifetime_range': (30, 40),
            'colors': [(128, 128, 128),(128, 128, 5),(128, 128, 5)]
        }
        self.is_exploding = False
        self.explosion_particles = []
    
    def is_active(self) -> bool:
        return self.alive or self.is_exploding
    
    def emit(self, screen):
        if self.alive:
            self.create_particles()
        self.update_particles()
        for p in self.particles:
            pygame.draw.circle(screen, p['color'], p['pos'].astype(int), int(p['size'] * (p['lifetime'] / p['max_lifetime'])))
            
    def ignite(self):
        self.is_exploding = True
        for _ in range(self.explosion_config['particle_count']):
            angle       = random.uniform(0, 360)
            speed       = random.uniform(*self.explosion_config['speed_range'])
            size        = random.uniform(*self.explosion_config['size_range'])
            lifetime    = random.randint(*self.explosion_config['lifetime_range'])
            color       = random.choice(self.explosion_config['colors'])
            velocity    = np.array([np.cos(np.radians(angle)) * speed, np.sin(np.radians(angle)) * speed])
            self.explosion_particles.append({
                'pos': self.pos.copy(),
                'vel': velocity,
                'size': size,
                'color': color,
                'lifetime': lifetime,
                'max_lifetime': lifetime
            })
    
    def update_explosion(self):
        if not self.is_exploding:
            return
        updated_particles = []
        for p in self.explosion_particles:
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                continue
            p['pos'] += p['vel']
            p['vel'][1] += self.config.GRAVITY * 0.2
            fade = p['lifetime'] / p['max_lifetime']
            p['color'] = tuple(255-int(c * fade) for c in p['color'])
            p['size'] *= 0.98
            updated_particles.append(p)
        if len(updated_particles) > 0:
            self.explosion_particles = updated_particles
            return
        self.is_exploding = False
        
    def create_particles(self):
        for _ in range(self.trail_config['emission_rate']):
            size        = random.uniform(*self.trail_config['size_range'])
            lifetime    = random.randint(*self.trail_config['lifetime_range'])
            speed       = random.uniform(*self.trail_config['speed_range'])
            angle       = random.uniform(-self.trail_config['spread_angle'], self.trail_config['spread_angle'])
            direction   = np.array([np.cos(np.radians(angle)) * speed, -np.sin(np.radians(angle)) * speed])
            color = self.get_interpolated_color(size) # interpolate color
            self.particles.append({
                'pos': self.pos.copy(),
                'vel': direction,
                'size': size,
                'color': color,
                'lifetime': lifetime,
                'max_lifetime': lifetime
            })
    
    def get_interpolated_color(self, size):
        color_map = self.trail_config['color_map']
        sizes = sorted(color_map.keys())
        # handle edge cases
        if size <= sizes[0]: return color_map[sizes[0]]
        if size >= sizes[-1]: return color_map[sizes[-1]]
        # find bracketing sizes
        for i in range(len(sizes) - 1):
            if sizes[i] <= size <= sizes[i + 1]:
                t = (size - sizes[i]) / (sizes[i + 1] - sizes[i])
                c1 = np.array(color_map[sizes[i]])
                c2 = np.array(color_map[sizes[i + 1]])
                return tuple(map(int, c1 + t * (c2 - c1)))

    def update_particles(self):
        updated_particles = []
        for p in self.particles:
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                continue
            p['pos'] += p['vel']
            p['vel'][1] += self.config.GRAVITY * 0.3
            fade = p['lifetime'] / p['max_lifetime']
            p['color'] = tuple(int(c * fade) for c in p['color'])
            updated_particles.append(p)
        self.particles = updated_particles

    def update(self, terrain, obstacles: List[pygame.Rect]) -> None:
        # explode if uncaught
        if not self.alive and not self.is_exploding:
            self.ignite()
            return
            
        # after effect
        if self.is_exploding:
            self.update_explosion()
            return
            
        # Update physics
        self.vel[1] += self.config.GRAVITY
        self.vel *= (1 - self.config.AIR_RESISTANCE)
        self.pos += self.vel
        self.trajectory.append(self.pos.copy())
        self.flight_time += 1
        
        # check obstacles
        for obstacle in obstacles:
            if obstacle.collidepoint(*self.pos):
                self.hitdata = 'obstacle'
                self.alive = False
                self.contact = self.pos.copy()
                self.ignite()
                return
                
        # check terrain
        if terrain.check_collision(self.pos):
            terrain.apply_explosion(self.pos, random.randrange(10,30))
            self.hitdata = 'terrain'
            self.alive = False
            self.contact = self.pos.copy()
            self.ignite()
            return
            
        # check bounds
        if (self.pos[0] < 0 or 
            self.pos[0] > self.config.WINDOW_SIZE[0] or
            self.pos[1] > self.config.WINDOW_SIZE[1]):
            self.hitdata = 'bounds'
            self.alive = False
            self.contact = self.pos.copy()
            self.ignite()
            return
            
        # check lifetime
        self.lifetime += 1
        if self.lifetime > self.lifetime_m:
            self.hitdata = 'despawn'
            self.alive = False
            self.ignite()
            self.contact = self.pos.copy()
            return
            
    def draw(self, screen: pygame.Surface) -> None:
        if self.alive and not self.is_exploding:
            self.emit(screen)
            if len(self.trajectory) > 1:
                pygame.draw.lines(screen, self.config.COLORS['GRAY'], False, [p.astype(int) for p in self.trajectory], 1)
            pygame.draw.circle(screen, self.config.COLORS['GRAY'], self.pos.astype(int), self.radius)    
        if not self.alive and self.is_exploding:
            for p in self.explosion_particles:
                #pygame.draw.circle(screen, p['color'], p['pos'].astype(int), int(p['size']))
                pygame.draw.rect(screen, p['color'], (p['pos'][0], p['pos'][1], int(p['size']), int(p['size'])))
                
    def get_metrics(self) -> dict:
        if len(self.trajectory) < 3 or self.contact is None:
            return None
        path = np.array(self.trajectory)
        # Find peak point' ('y' is inverted in pygame)
        peak_idx = np.argmin(path[:, 1])
        peak_point = path[peak_idx]
        # Calculate peak angle (angle to highest point)
        peak_dx = peak_point[0] - self.origin[0]
        # Invert Y for correct angle
        peak_dy = self.origin[1] - peak_point[1]
        peak_angle = np.degrees(np.arctan2(peak_dy, peak_dx))
        # Calculate landing angle if projectile has landed
        landing_angle = None
        if self.contact is not None:
            # Use last few points to determine landing angle
            last_points = path[-min(len(path), 5):]
            if len(last_points) >= 2:
                dx = last_points[-1][0] - last_points[0][0]
                dy = last_points[-1][1] - last_points[0][1]
                landing_angle = np.degrees(np.arctan2(dy, dx))
        # Calculate average angle along the path
        angles = []
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i-1][1] - path[i][1]  # Invert Y for correct angle
            if dx != 0 or dy != 0:  # Avoid division by zero
                angles.append(np.degrees(np.arctan2(dy, dx)))
        average_angle = np.mean(angles) if angles else None
        # Calculate arc length (total path length)
        arc_length = 0
        for i in range(1, len(path)):
            arc_length += np.linalg.norm(path[i] - path[i-1])
        # Calculate total horizontal distance
        total_distance = path[-1][0] - path[0][0]
        # Calculate peak height (relative to start)
        peak_height = self.origin[1] - peak_point[1]  # Invert Y for correct height

        return {
            'peak_angle': peak_angle,
            'landing_angle': landing_angle,
            'average_angle': average_angle,
            'arc_length': arc_length,
            'total_distance': total_distance,
            'peak_height': peak_height,
            'distance': np.linalg.norm(self.contact - self.origin),
            'time': self.flight_time,
            'altitude': min(p[1] for p in self.trajectory),
            'contact': self.contact.copy(),
            'hitdata': self.hitdata
        }

class Terrain:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.surface = pygame.Surface((width, height))
        # transparent background
        self.surface.fill((135, 206, 235))  # Sky blue
        self.ground_color = (139, 69, 19)   # Brown
        self.height_map = []
        # empty terrain
        self.generate_terrain()
        # collision mask
        self.mask = pygame.mask.from_surface(self.surface)
        
    def generate_terrain(self):
        # allocate surface
        self.surface.fill((135, 206, 235))
        
        # height map
        self.height_map = []
        y = self.height * 0.7  # Start at 70% of screen height
        
        # smoothing using interpolation
        control_points = []
        num_control_points = 10
        for i in range(num_control_points):
            x = i * (self.width / (num_control_points - 1))
            y = self.height * 0.7 + random.randint(-50, 50)
            control_points.append((x, y))
            
        # interpolate
        for x in range(self.width):
            # find surrounding
            for i in range(len(control_points) - 1):
                if control_points[i][0] <= x <= control_points[i + 1][0]:
                    # lerp
                    x1, y1 = control_points[i]
                    x2, y2 = control_points[i + 1]
                    t = (x - x1) / (x2 - x1)
                    y = y1 + t * (y2 - y1)
                    self.height_map.append(int(y))
                    break
        
        # create terrain polygon
        # add top terrain points
        terrain_points = []
        for x in range(self.width):
            terrain_points.append((x, self.height_map[x]))
        # add bottom corners to complete the polygon
        terrain_points.append((self.width, self.height))  # bottom right
        terrain_points.append((0, self.height))           # bottom left
        
        # Draw the terrain
        pygame.draw.polygon(self.surface, self.ground_color, terrain_points)

    def create_explosion_mask(self, center: tuple, radius: int) -> pygame.mask.Mask:
        # Create circular explosion mask
        explosion_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        explosion_surf.fill((0, 0, 0, 0))  # Make sure surface is transparent
        pygame.draw.circle(explosion_surf, (255, 255, 255, 255), (radius, radius), radius)
        return pygame.mask.from_surface(explosion_surf)

    def apply_explosion(self, pos: tuple, radius: int):
        x, y = int(pos[0] - radius), int(pos[1] - radius)
        temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, (135, 206, 235), (radius, radius), radius)
        self.surface.blit(temp_surf, (x, y))
        self.mask = pygame.mask.from_surface(self.surface)
        return

    def check_collision(self, pos: tuple) -> bool:
        x, y = int(pos[0]), int(pos[1])
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        try:
            return self.surface.get_at((x, y)) == self.ground_color
        except IndexError:
            return False

    def draw(self, screen: pygame.Surface):
        screen.blit(self.surface, (0, 0))
