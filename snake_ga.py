# 首先安装必要的依赖:
# pip install gymnasium numpy torch stable-baselines3
# 可选依赖:
# pip install tensorboard  # 用于训练可视化

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
import os

class SnakeEnv(gym.Env):
    # [保持 SnakeEnv 类的实现不变]
    def __init__(self, width=600, height=400, grid_size=20):
        super().__init__()
        self.width = width
        self.height = height
        self.grid_size = grid_size
        
        self.action_space = spaces.Discrete(4)
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
        self.max_steps = 200
        return self._get_state(), {}
    
    def _new_food(self):
        while True:
            x = self.np_random.integers(0, self.width//self.grid_size) * self.grid_size
            y = self.np_random.integers(0, self.height//self.grid_size) * self.grid_size
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
            self.steps / self.max_steps
        ], dtype=np.float32)
        
        return state
    
    def _check_collision(self, x, y):
        return 1.0 if (x < 0 or x >= self.width or 
                      y < 0 or y >= self.height or 
                      (x, y) in self.snake) else 0.0
    
    def step(self, action):
        self.steps += 1
        terminated = False
        
        current_dx, current_dy = self.dx, self.dy
        if action == 0 and current_dy != self.grid_size:  # 上
            self.dx, self.dy = 0, -self.grid_size
        elif action == 1 and current_dx != -self.grid_size:  # 右
            self.dx, self.dy = self.grid_size, 0
        elif action == 2 and current_dy != -self.grid_size:  # 下
            self.dx, self.dy = 0, self.grid_size
        elif action == 3 and current_dx != self.grid_size:  # 左
            self.dx, self.dy = -self.grid_size, 0
        
        new_head = (self.snake[0][0] + self.dx, self.snake[0][1] + self.dy)
        
        if (new_head in self.snake or 
            new_head[0] < 0 or new_head[0] >= self.width or 
            new_head[1] < 0 or new_head[1] >= self.height):
            terminated = True
            reward = -1
            return self._get_state(), reward, terminated, False, {'score': self.score}
        
        self.snake.insert(0, new_head)
        
        reward = 0.1
        
        if self.snake[0] == self.food:
            self.score += 1
            self.food = self._new_food()
            reward = 1.0 + (len(self.snake) * 0.1)
        else:
            self.snake.pop()
        
        if self.steps >= self.max_steps:
            terminated = True
            reward = -0.5
        
        return self._get_state(), reward, terminated, False, {'score': self.score}

def make_env():
    def _init():
        env = SnakeEnv()
        return env
    return _init

def train_snake(total_timesteps=1000000):
    # 创建日志目录
    log_dir = "./logs"
    model_dir = "./models"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 创建向量化环境
    env = DummyVecEnv([make_env()])
    env = VecMonitor(env)
    
    # 创建评估环境
    eval_env = DummyVecEnv([make_env()])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    # 检查是否安装了tensorboard
    try:
        import tensorboard
        tensorboard_log = "./snake_tensorboard/"
    except ImportError:
        print("TensorBoard not installed. Training will proceed without TensorBoard logging.")
        tensorboard_log = None
    
    # 创建并训练模型
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log=tensorboard_log
    )
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
    finally:
        # 保存最终模型
        model.save(os.path.join(model_dir, "snake_final_model"))
    
    return model

if __name__ == "__main__":
    print("Starting Snake AI training...")
    print("Required packages: gymnasium, numpy, torch, stable-baselines3")
    print("Optional package: tensorboard (for training visualization)")
    print("\nPress Ctrl+C to stop training and save the model")
    print("=" * 50)
    
    model = train_snake()
