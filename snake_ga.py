# 首先安装必要的依赖:
# pip install gymnasium numpy torch stable-baselines3
# 可选依赖:
# pip install tensorboard  # 用于训练可视化

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor,SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed
import os
import multiprocessing

class SnakeEnv(gym.Env):
    # [保持 SnakeEnv 类的实现不变]
    def __init__(self, width=600, height=400, grid_size=20):
        super().__init__()
        self.width = width
        self.height = height
        self.grid_size = grid_size
        
        self.action_space = spaces.Discrete(4)
        # 扩展状态空间
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(25,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.snake = [(self.width//2, self.height//2)]
        self.dx, self.dy = self.grid_size, 0
        self.food = self._new_food()
        self.score = 0
        self.steps = 0
        self.max_steps = max(12000, len(self.snake) * 200)
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
        
        # 计算到食物的距离和方向
        food_distance = ((food_x - head_x)**2 + (food_y - head_y)**2)**0.5
        food_direction = [
            float(food_x < head_x),
            float(food_x > head_x),
            float(food_y < head_y),
            float(food_y > head_y)
        ]
        
        # 计算到最近墙壁的距离
        wall_distances = [
            head_x / self.width,  # 左墙
            (self.width - head_x) / self.width,  # 右墙
            head_y / self.height,  # 上墙
            (self.height - head_y) / self.height  # 下墙
        ]
        
        # 计算蛇头方向与食物的夹角
        angle = 0.0
        if self.dx != 0 or self.dy != 0:
            current_direction = np.array([self.dx, self.dy])
            food_vector = np.array([food_x - head_x, food_y - head_y])
            if np.linalg.norm(food_vector) > 0:
                angle = np.dot(current_direction, food_vector) / (np.linalg.norm(current_direction) * np.linalg.norm(food_vector))
        
        # 危险区域检测
        danger_map = []
        for dy in [-self.grid_size, 0, self.grid_size]:
            for dx in [-self.grid_size, 0, self.grid_size]:
                if dx == 0 and dy == 0:
                    continue
                danger_map.append(self._check_collision(head_x + dx, head_y + dy))
        
        # 计算空间利用率
        total_cells = (self.width // self.grid_size) * (self.height // self.grid_size)
        space_utilization = len(self.snake) / total_cells
        
        state = np.array([
            (food_x - head_x) / self.width,
            (food_y - head_y) / self.height,
            food_distance / (self.width**2 + self.height**2)**0.5,
            self.dx / self.grid_size,
            self.dy / self.grid_size,
            angle,  # 添加角度信息
            *food_direction,
            *wall_distances,  # 添加墙壁距离
            *danger_map,
            space_utilization,  # 添加空间利用率
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
        
        # 更新方向
        if action == 0 and current_dy != self.grid_size:  # 上
            self.dx, self.dy = 0, -self.grid_size
        elif action == 1 and current_dx != -self.grid_size:  # 右
            self.dx, self.dy = self.grid_size, 0
        elif action == 2 and current_dy != -self.grid_size:  # 下
            self.dx, self.dy = 0, self.grid_size
        elif action == 3 and current_dx != self.grid_size:  # 左
            self.dx, self.dy = -self.grid_size, 0
        
        new_head = (self.snake[0][0] + self.dx, self.snake[0][1] + self.dy)
        
        # 碰撞检测
        if (new_head in self.snake or 
            new_head[0] < 0 or new_head[0] >= self.width or 
            new_head[1] < 0 or new_head[1] >= self.height):
            terminated = True
            reward = -2.0  # 增加碰撞惩罚
            return self._get_state(), reward, terminated, False, {'score': self.score}
        
        # 计算移动前后到食物的距离变化
        old_distance = ((self.snake[0][0] - self.food[0])**2 + 
                       (self.snake[0][1] - self.food[1])**2)**0.5
        new_distance = ((new_head[0] - self.food[0])**2 + 
                       (new_head[1] - self.food[1])**2)**0.5
        
        # 计算到墙壁的距离
        wall_distance = min(
            new_head[0],  # 左墙
            self.width - new_head[0],  # 右墙
            new_head[1],  # 上墙
            self.height - new_head[1]  # 下墙
        ) / self.grid_size
        
        # 根据距离变化和墙壁距离计算奖励
        distance_reward = (old_distance - new_distance) * 0.02
        wall_penalty = -0.01 if wall_distance <= 1 else 0  # 靠近墙壁的惩罚
        
        # 添加新的头部
        self.snake.insert(0, new_head)
        
        # 基础奖励
        reward = 0.01 + distance_reward + wall_penalty
        
        # 吃到食物的奖励
        if self.snake[0] == self.food:
            self.score += 1
            self.food = self._new_food()
            # 根据蛇长度和剩余空间动态调整奖励
            space_left = 1 - (len(self.snake) / ((self.width // self.grid_size) * (self.height // self.grid_size)))
            reward = 3.0 + (len(self.snake) * 0.3) * (1 / space_left)
        else:
            self.snake.pop()
        
        # 检查重复移动
        if not hasattr(self, 'position_history'):
            self.position_history = []
        
        self.position_history.append(new_head)
        if len(self.position_history) > 50:
            self.position_history.pop(0)
            unique_positions = len(set(self.position_history))
            if unique_positions < 10:
                reward -= 1.0  # 增加重复移动惩罚
        
        # 检查最大步数
        if self.steps >= self.max_steps:
            terminated = True
            reward = -1.0
        
        return self._get_state(), reward, terminated, False, {'score': self.score}

def make_env(rank, seed=0):
    """
    创建环境的工厂函数，用于并行环境
    """
    def _init():
        env = SnakeEnv()
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init

def train_snake(total_timesteps=10000000):
    # 创建日志和模型目录
    log_dir = "./logs"
    model_dir = "./models"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 确定可用的 CPU 核心数，留出一个核心给系统
    num_cpu = max(1, multiprocessing.cpu_count() - 1)
    
    # 创建并行环境
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    env = VecMonitor(env)
    
    # 创建评估环境（单个环境即可）
    eval_env = SubprocVecEnv([make_env(num_cpu)])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    # 检查是否安装了 tensorboard
    try:
        import tensorboard
        tensorboard_log = "./snake_tensorboard/"
    except ImportError:
        print("TensorBoard not installed. Training will proceed without TensorBoard logging.")
        tensorboard_log = None
    
    # 优化批处理大小和步数以适应并行环境
    n_steps = 2048 // num_cpu  # 确保总步数保持不变
    batch_size = min(64 * num_cpu, n_steps * num_cpu)  # 根据并行度调整批处理大小
    
    # 创建并训练模型
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=2e-4,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=12,
        gamma=0.995,
        gae_lambda=0.98,
        clip_range=0.15,
        ent_coef=0.01,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[512, 384, 256, 192, 128],  # 更深的策略网络
                vf=[512, 384, 256, 192, 128]   # 更深的价值网络
            )
        ),
        tensorboard_log=tensorboard_log,
        device='cpu'  # 确保使用 CPU 训练
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
        # 清理环境
        env.close()
        eval_env.close()
    
    return model

if __name__ == "__main__":
    print(f"Starting Snake AI training using {max(1, multiprocessing.cpu_count() - 1)} CPU cores...")
    print("Required packages: gymnasium, numpy, torch, stable-baselines3")
    print("Optional package: tensorboard (for training visualization)")
    print("\nPress Ctrl+C to stop training and save the model")
    print("=" * 50)
    
    # 设置全局随机种子
    set_random_seed(42)
    
    model = train_snake()
