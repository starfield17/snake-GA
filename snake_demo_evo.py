import pygame
import sys
import torch
from snake_ga import NeuralNetwork, SnakeEnv
import glob
import os
import time
import colorsys

pygame.init()

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (20, 20, 20)
RED = (255, 50, 50)
GREEN = (50, 255, 100)
BLUE = (50, 150, 255)
GRAY = (40, 40, 40)
YELLOW = (255, 255, 0)

WIDTH = 600
HEIGHT = 400
GRID_SIZE = 20
SPEED = 30

class SnakeGame:
    def __init__(self):
        # 设置窗口和标题
        self.screen = pygame.display.set_mode((WIDTH + 200, HEIGHT))  # 增加了200像素宽度用于显示信息
        pygame.display.set_caption("Snake AI Visualization")
        self.clock = pygame.time.Clock()
        self.env = SnakeEnv(WIDTH, HEIGHT, GRID_SIZE)
        
        # 加载字体
        try:
            self.font = pygame.font.Font(None, 32)
            self.small_font = pygame.font.Font(None, 24)
        except:
            print("Error loading fonts")
            sys.exit(1)
        
        # 初始化变量
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0
        self.action_probs = torch.zeros(4)  # 存储动作概率
        
        # 加载模型
        self.model = NeuralNetwork()
        self._load_best_model()
        
        self.games_played = 0
        self.total_score = 0
        self.best_score = 0
        self.last_10_scores = []
        
    def _load_best_model(self):
        model_files = glob.glob("snake_best_*.pth")
        if not model_files:
            print("No model files found.")
            sys.exit(1)
            
        best_model_path = max(model_files, key=lambda x: float(x.split('_')[-1].split('.')[0]))
        try:
            self.model.load_state_dict(torch.load(best_model_path))
            self.model.eval()
            print(f"Loaded model: {best_model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

    def _get_gradient_color(self, start_color, end_color, ratio):
        return tuple(int(start + (end - start) * ratio) for start, end in zip(start_color, end_color))

    def draw_grid(self):
        for x in range(0, WIDTH, GRID_SIZE):
            alpha = 0.3 + 0.1 * (x / WIDTH)  # 渐变效果
            color = tuple(int(c * alpha) for c in GRAY)
            pygame.draw.line(self.screen, color, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, GRID_SIZE):
            alpha = 0.3 + 0.1 * (y / HEIGHT)
            color = tuple(int(c * alpha) for c in GRAY)
            pygame.draw.line(self.screen, color, (0, y), (WIDTH, y))
    
    def draw_snake(self):
        # 绘制蛇身
        for i, segment in enumerate(self.env.snake[1:]):
            alpha = 0.5 + 0.5 * (i / len(self.env.snake))  # 渐变效果
            color = self._get_gradient_color(GREEN, (30, 200, 70), alpha)
            pygame.draw.rect(self.screen, color,
                           (segment[0], segment[1], GRID_SIZE-2, GRID_SIZE-2), border_radius=3)
        
        # 绘制蛇头
        head = self.env.snake[0]
        pygame.draw.rect(self.screen, BLUE,
                        (head[0], head[1], GRID_SIZE-2, GRID_SIZE-2), border_radius=4)
        
        # 添加蛇头方向指示器
        direction_indicator = (head[0] + GRID_SIZE//2, head[1] + GRID_SIZE//2)
        pygame.draw.circle(self.screen, WHITE, direction_indicator, 3)
    
    def draw_food(self):
        # 脉动效果
        pulse = (1 + 0.2 * abs(pygame.time.get_ticks() % 1000 - 500) / 500)
        size = int(GRID_SIZE * pulse)
        x = self.env.food[0] + (GRID_SIZE - size) // 2
        y = self.env.food[1] + (GRID_SIZE - size) // 2
        pygame.draw.rect(self.screen, RED, (x, y, size-2, size-2), border_radius=5)
    
    def draw_stats(self):
        # 绘制右侧信息面板
        panel_x = WIDTH + 10
        pygame.draw.rect(self.screen, GRAY, (WIDTH, 0, 200, HEIGHT))
        
        stats = [
            ("Score", f"{self.env.score}"),
            ("Best", f"{self.best_score}"),
            ("Games", f"{self.games_played}"),
            ("Avg", f"{self.total_score/max(1,self.games_played):.1f}"),
            ("Steps", f"{self.env.steps}"),
            ("FPS", f"{self.fps}")
        ]
        
        for i, (label, value) in enumerate(stats):
            # 标签
            label_surface = self.small_font.render(label + ":", True, (200, 200, 200))
            self.screen.blit(label_surface, (panel_x, 20 + i*40))
            # 值
            value_surface = self.font.render(value, True, WHITE)
            self.screen.blit(value_surface, (panel_x + 70, 15 + i*40))

        # 绘制动作概率条
        self.draw_action_bars(panel_x, 280)
        
        # 最近分数趋势
        if self.last_10_scores:
            self.draw_score_trend(panel_x, 350)
    
    def draw_action_bars(self, x, y):
        actions = ["up", "right", "down", "left"]
        max_width = 150
        
        title = self.small_font.render("Action Probabilities:", True, (200, 200, 200))
        self.screen.blit(title, (x, y - 20))
        
        for i, (action, prob) in enumerate(zip(actions, self.action_probs)):
            prob = float(prob)
            bar_width = int(prob * max_width)
            color = self._get_gradient_color((50, 50, 200), (100, 200, 255), prob)
            
            pygame.draw.rect(self.screen, GRAY, (x, y + i*25, max_width, 20))
            pygame.draw.rect(self.screen, color, (x, y + i*25, bar_width, 20))
            
            label = self.small_font.render(f"{action} {prob:.2f}", True, WHITE)
            self.screen.blit(label, (x + 5, y + i*25 + 2))
    
    def draw_score_trend(self, x, y):
        if len(self.last_10_scores) < 2:
            return
            
        max_score = max(self.last_10_scores)
        min_score = min(self.last_10_scores)
        range_score = max(1, max_score - min_score)
        
        points = []
        for i, score in enumerate(self.last_10_scores):
            px = x + (i * 180 // len(self.last_10_scores))
            py = y + 40 - int((score - min_score) * 40 / range_score)
            points.append((px, py))
            
        if len(points) > 1:
            pygame.draw.lines(self.screen, YELLOW, False, points, 2)
    
    def run(self):
        global SPEED
        running = True
        state = self.env.reset()
        last_time = time.time()
        
        while running:
            current_time = time.time()
            self.frame_count += 1
            if current_time - last_time > 1:
                self.fps = self.frame_count
                self.frame_count = 0
                last_time = current_time
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        state = self.env.reset()
                    elif event.key == pygame.K_UP:
                        SPEED = min(100, SPEED + 10)
                    elif event.key == pygame.K_DOWN:
                        SPEED = max(10, SPEED - 10)
            
            # AI决策
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                self.action_probs = self.model(state_tensor)[0]
            action = self.action_probs.argmax().item()
            
            # 环境更新
            state, _ = self.env.step(action)
            
            if self.env.game_over:
                self.games_played += 1
                self.total_score += self.env.score
                self.best_score = max(self.best_score, self.env.score)
                self.last_10_scores.append(self.env.score)
                if len(self.last_10_scores) > 10:
                    self.last_10_scores.pop(0)
                state = self.env.reset()
            
            # 渲染
            self.screen.fill(WHITE)
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
