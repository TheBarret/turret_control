import numpy as np
import pygame
import uuid

from dataclasses import dataclass
from typing import List, Tuple, Optional

from data import Projectile, Terrain, Config

class Turret:
    def __init__(self, pos: np.ndarray, config: Config):
        self.id = uuid.uuid4()
        self.config = config
        self.pos = pos
        self.ready = True
        self.fired = 0
        self.angle = 50.0
        self.power = 11.0
        self.projectile = None
        
    def adjust_angle(self, delta: float) -> None:
        self.angle = np.clip(self.angle + delta, self.config.MIN_ANGLE, self.config.MAX_ANGLE)
    
    def adjust_power(self, delta: float) -> None:
        self.power = np.clip(self.power + delta, self.config.MIN_POWER, self.config.MAX_POWER)
    
    def fire(self) -> bool:
        if not self.ready or self.projectile is not None:
            return False
            
        barrel_end = self.get_barrel_end()
        self.projectile = Projectile(barrel_end, self.angle, self.power, self.config)
        self.ready = False
        self.fired += 1
        return True
    
    def get_barrel_end(self) -> np.ndarray:
        return self.pos + np.array([
            np.cos(np.radians(self.angle)) * self.config.BARREL_LENGTH,
            -np.sin(np.radians(self.angle)) * self.config.BARREL_LENGTH
        ])
    
    def update(self, terrain, obstacles: List[pygame.Rect]) -> Optional[dict]:
        metrics = None
        if self.projectile:
            self.projectile.update(terrain , obstacles)
            if not self.projectile.is_active():
                metrics = self.projectile.get_metrics()
                self.projectile = None
                self.ready = True
        return metrics
    
    def draw(self, screen: pygame.Surface, font: Optional[pygame.font.Font] = None) -> None:
        # Draw barrel
        barrel_end = self.get_barrel_end()
        pygame.draw.line(screen, self.config.COLORS['GRAY'], self.pos.astype(int), barrel_end.astype(int), 3)
        # Draw base
        pygame.draw.circle(screen, self.config.COLORS['RED'], self.pos.astype(int), 8)
        # Draw projectile if exists
        if self.projectile:
            self.projectile.draw(screen)
        
        # Draw stats if font provided
        if font:
            angle_text = font.render(f'Angle: {self.angle:.1f}°', True, self.config.COLORS['WHITE'])
            power_text = font.render(f'Power: {self.power:.1f}', True, self.config.COLORS['WHITE'])
            screen.blit(angle_text, (10, 10))
            screen.blit(power_text, (10, 30))

class Game:
    def __init__(self, config: Config = Config()):
        pygame.init()
        self.config = config
        self.screen = pygame.display.set_mode(config.WINDOW_SIZE)
        pygame.display.set_caption("Turret Training Ground")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        # create terrain
        self.terrain = Terrain(config.WINDOW_SIZE[0], config.WINDOW_SIZE[1])
        # create turret at starting position
        start_pos = np.array([50, config.WINDOW_SIZE[1] / 2  - 50], dtype=float)
        self.turret = Turret(start_pos, config)
        # create obstacles
        self.obstacles = [pygame.Rect(400, 200, 50, 400)]
        
    def handle_input(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.turret.fire()
                elif event.key == pygame.K_LEFT:
                    self.turret.adjust_angle(self.config.ANGLE_STEP)
                elif event.key == pygame.K_RIGHT:
                    self.turret.adjust_angle(-self.config.ANGLE_STEP)
                elif event.key == pygame.K_UP:
                    self.turret.adjust_power(self.config.POWER_STEP)
                elif event.key == pygame.K_DOWN:
                    self.turret.adjust_power(-self.config.POWER_STEP)
        return True
    
    def run(self) -> None:
        running = True
        while running:
            # Handle input
            running = self.handle_input()
            self.screen.fill(self.config.COLORS['BLACK'])

            # draw terrain
            self.terrain.draw(self.screen)
            
            # draw obstacles
            if self.obstacles:
                for obstacle in self.obstacles:
                    pygame.draw.rect(self.screen, self.config.COLORS['GRAY'], obstacle)
            
            # update game objects
            metrics = self.turret.update(self.terrain, self.obstacles)
            if metrics:
                annotation = [f"{key}: {value}" for key, value in metrics.items()]
                print("--[Projectile Metrics]------------------------------------")
                print("\n".join(annotation))
      
                    
            # draw turret
            self.turret.draw(self.screen, self.font)
            
            pygame.display.flip()
            self.clock.tick(self.config.FPS)
        
        pygame.quit()

if __name__ == '__main__':
    game = Game()
    game.run()