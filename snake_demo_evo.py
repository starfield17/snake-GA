import pygame
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from stable_baselines3 import PPO
import time
import os
# 保持与训练代码相同的环境设置
WIDTH = 600
HEIGHT = 400
GRID_SIZE = 20
SPEED = 30

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (20, 20, 20)
RED = (255, 50, 50)
GREEN = (50, 255, 100)
BLUE = (50, 150, 255)
GRAY = (40, 40, 40)
YELLOW = (255, 255, 0)

pygame.init()
class SnakeEnvDemo(gym.Env):
    def __init__(self, width=600, height=400, grid_size=20):
        super().__init__()
        self.width = width
        self.height = height
        self.grid_size = grid_size
        
        # 定义动作空间 (上、右、下、左)
        self.action_space = spaces.Discrete(4)
        
        # 定义观察空间 (与训练时相同的19个状态值)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(19,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.snake = [(self.width//2, self.height//2)]
        self.dx, self.dy = self.grid_size, 0
        self.food = self._new_food()
        self.score = 0
        self.steps = 0
        return self._get_state(), {}
    
    def _new_food(self):
        while True:
            x = np.random.randint(0, self.width//self.grid_size) * self.grid_size
            y = np.random.randint(0, self.height//self.grid_size) * self.grid_size
            if (x, y) not in self.snake:
                return (x, y)
    
    def _get_state(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        
        food_distance = ((food_x - head_x)**2 + (food_y - head_y)**2)**0.5
        food_direction = [
            float(food_x < head_x),
            float(food_x > head_x),
            float(food_y < head_y),
            float(food_y > head_y)
        ]
        
        danger_map = []
        for dy in [-self.grid_size, 0, self.grid_size]:
            for dx in [-self.grid_size, 0, self.grid_size]:
                if dx == 0 and dy == 0:
                    continue
                danger_map.append(self._check_collision(head_x + dx, head_y + dy))
        
        state = np.array([
            (food_x - head_x) / self.width,
            (food_y - head_y) / self.height,
            food_distance / (self.width**2 + self.height**2)**0.5,
            self.dx / self.grid_size,
            self.dy / self.grid_size,
            *food_direction,
            *danger_map,
            len(self.snake) / (self.width * self.height / (self.grid_size * self.grid_size)),
            self.steps / 1000.0  # 使用一个大数来归一化步数
        ], dtype=np.float32)
        
        return state
    
    def _check_collision(self, x, y):
        return 1.0 if (x < 0 or x >= self.width or 
                      y < 0 or y >= self.height or 
                      (x, y) in self.snake) else 0.0
    
    def step(self, action):
        self.steps += 1
        terminated = False
        
        # 更新蛇的方向
        current_dx, current_dy = self.dx, self.dy
        if action == 0 and current_dy != self.grid_size:  # 上
            self.dx, self.dy = 0, -self.grid_size
        elif action == 1 and current_dx != -self.grid_size:  # 右
            self.dx, self.dy = self.grid_size, 0
        elif action == 2 and current_dy != -self.grid_size:  # 下
            self.dx, self.dy = 0, self.grid_size
        elif action == 3 and current_dx != self.grid_size:  # 左
            self.dx, self.dy = -self.grid_size, 0
        
        # 移动蛇头
        new_head = (self.snake[0][0] + self.dx, self.snake[0][1] + self.dy)
        
        # 检查碰撞
        if (new_head in self.snake or 
            new_head[0] < 0 or new_head[0] >= self.width or 
            new_head[1] < 0 or new_head[1] >= self.height):
            terminated = True
            reward = -1
            return self._get_state(), reward, terminated, False, {'score': self.score}
        
        self.snake.insert(0, new_head)
        
        # 基础奖励
        reward = 0.1
        
        # 吃到食物
        if self.snake[0] == self.food:
            self.score += 1
            self.food = self._new_food()
            reward = 1.0 + (len(self.snake) * 0.1)
        else:
            self.snake.pop()
        
        return self._get_state(), reward, terminated, False, {'score': self.score}
class SnakeGame:
    def __init__(self):
        # 设置窗口和标题
        self.screen = pygame.display.set_mode((WIDTH + 200, HEIGHT))
        pygame.display.set_caption("Snake AI Visualization (PPO)")
        self.clock = pygame.time.Clock()
        
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
        self.action_probs = np.zeros(4)
        
        # 加载模型
        self._load_model()
        
        self.games_played = 0
        self.total_score = 0
        self.best_score = 0
        self.last_10_scores = []
        
        # 创建演示环境（使用无步数限制的版本）
        self.env = SnakeEnvDemo(WIDTH, HEIGHT, GRID_SIZE)
    
    def _load_model(self):
        model_path = "./models/snake_final_model.zip"
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            sys.exit(1)
            
        try:
            self.model = PPO.load(model_path)
            print(f"Loaded model: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

    def _get_gradient_color(self, start_color, end_color, ratio):
        return tuple(int(start + (end - start) * ratio) for start, end in zip(start_color, end_color))

    def draw_grid(self):
        for x in range(0, WIDTH, GRID_SIZE):
            alpha = 0.3 + 0.1 * (x / WIDTH)
            color = tuple(int(c * alpha) for c in GRAY)
            pygame.draw.line(self.screen, color, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, GRID_SIZE):
            alpha = 0.3 + 0.1 * (y / HEIGHT)
            color = tuple(int(c * alpha) for c in GRAY)
            pygame.draw.line(self.screen, color, (0, y), (WIDTH, y))
    
    def draw_snake(self):
        # 绘制蛇身
        for i, segment in enumerate(self.env.snake[1:]):
            alpha = 0.5 + 0.5 * (i / len(self.env.snake))
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
            label_surface = self.small_font.render(label + ":", True, (200, 200, 200))
            self.screen.blit(label_surface, (panel_x, 20 + i*40))
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
        state, _ = self.env.reset()
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
                        state, _ = self.env.reset()
                    elif event.key == pygame.K_UP:
                        SPEED = min(100, SPEED + 10)
                    elif event.key == pygame.K_DOWN:
                        SPEED = max(10, SPEED - 10)
            
            # AI决策
            action, _states = self.model.predict(state, deterministic=True)
            # 获取动作概率（这里简化处理，因为PPO不直接输出概率）
            self.action_probs = np.zeros(4)
            self.action_probs[action] = 1.0
            
            # 环境更新
            state, _, terminated, truncated, info = self.env.step(action)
            
            if terminated or truncated:
                self.games_played += 1
                self.total_score += self.env.score
                self.best_score = max(self.best_score, self.env.score)
                self.last_10_scores.append(self.env.score)
                if len(self.last_10_scores) > 10:
                    self.last_10_scores.pop(0)
                state, _ = self.env.reset()
            
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
    print("Controls:")
    print("Space: Reset game")
    print("Up/Down: Adjust speed")
    print("Esc: Quit")
    print("\nStarting visualization...")
    game = SnakeGame()
    game.run()
