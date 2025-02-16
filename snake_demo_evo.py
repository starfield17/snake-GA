import pygame
import sys
import torch
from snake_ga import NeuralNetwork, SnakeEnv
import glob
import os

pygame.init()

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


WIDTH = 600
HEIGHT = 400
GRID_SIZE = 20
SPEED = 10 

class SnakeGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Snake AI Demo")
        self.clock = pygame.time.Clock()
        self.env = SnakeEnv(WIDTH, HEIGHT, GRID_SIZE)
        
        # 加载模型
        self.model = NeuralNetwork()
        model_files = glob.glob("snake_ga_best_*.pth")
        
        if not model_files:
            print("No model files found.")
            sys.exit(1)
        
        best_model_path = None
        max_score = -float('inf')
        
        for file_path in model_files:
            filename = os.path.basename(file_path)
            parts = filename.split('_')
            
            if len(parts) < 4:
                continue
                
            score_part = parts[-1]
            score_str = os.path.splitext(score_part)[0]
            
            try:
                score = float(score_str)
                if score > max_score:
                    max_score = score
                    best_model_path = file_path
            except ValueError:
                continue  
        
        if not best_model_path:
            print("No valid model files found.")
            sys.exit(1)
            
        try:
            self.model.load_state_dict(torch.load(best_model_path))
            self.model.eval()
            print(f"Successfully loaded model from {best_model_path} (Score: {max_score})")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
        
        self.games_played = 0
        self.total_score = 0
        self.best_score = 0
        
    def draw_grid(self):
        for x in range(0, WIDTH, GRID_SIZE):
            pygame.draw.line(self.screen, (50, 50, 50), (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, GRID_SIZE):
            pygame.draw.line(self.screen, (50, 50, 50), (0, y), (WIDTH, y))
    
    def draw_snake(self):

        for segment in self.env.snake[1:]:
            pygame.draw.rect(self.screen, GREEN,
                           (segment[0], segment[1], GRID_SIZE-2, GRID_SIZE-2))
        
        head = self.env.snake[0]
        pygame.draw.rect(self.screen, BLUE,
                        (head[0], head[1], GRID_SIZE-2, GRID_SIZE-2))
    
    def draw_food(self):
        pygame.draw.rect(self.screen, RED,
                        (self.env.food[0], self.env.food[1], GRID_SIZE-2, GRID_SIZE-2))
    
    def draw_stats(self):
        font = pygame.font.Font(None, 36)
        stats = [
            f"Score: {self.env.score}",
            f"Games: {self.games_played}",
            f"Avg Score: {self.total_score/max(1,self.games_played):.2f}",
            f"Best Score: {self.best_score}",
            f"Steps: {self.env.steps}"
        ]
        
        for i, text in enumerate(stats):
            surface = font.render(text, True, WHITE)
            self.screen.blit(surface, (10, 10 + i*30))
    
    def run(self):
        running = True
        state = self.env.reset()
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        state = self.env.reset()
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_probs = self.model(state_tensor)
            action = action_probs.argmax().item()
            
            state, _ = self.env.step(action)
            
            if self.env.game_over:
                self.games_played += 1
                self.total_score += self.env.score
                self.best_score = max(self.best_score, self.env.score)
                state = self.env.reset()
            
            self.screen.fill(BLACK)
            self.draw_grid()
            self.draw_snake()
            self.draw_food()
            self.draw_stats()
            pygame.display.flip()

            self.clock.tick(SPEED)
        
        pygame.quit()

if __name__ == "__main__":
    game = SnakeGame()
    game.run()
