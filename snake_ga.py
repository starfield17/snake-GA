import numpy as np
import random
from typing import List, Tuple
import torch
import torch.nn as nn
import multiprocessing as mp
from functools import partial
import sys 
import copy
import resource
import os
WIDTH = 600
HEIGHT = 400
GRID_SIZE = 20

class NeuralNetwork(nn.Module):
    def __init__(self, input_size=19, hidden_size=128, output_size=4):
        super(NeuralNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
            nn.Softmax(dim=-1)
        )
        
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        return self.network(x)
    
    def get_weights(self) -> List[np.ndarray]:
        return [param.data.numpy() for param in self.parameters()]
    
    def set_weights(self, weights: List[np.ndarray]):
        for param, weight in zip(self.parameters(), weights):
            param.data = torch.FloatTensor(weight)

class SnakeEnv:
    def __init__(self, width, height, grid_size, max_steps_without_food=200):
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.max_steps_without_food = max_steps_without_food
        self.reset()
    
    def reset(self):
        self.snake = [(self.width//2, self.height//2)]
        self.dx, self.dy = self.grid_size, 0
        self.food = self._new_food()
        self.score = 0
        self.steps = 0
        self.steps_without_food = 0
        self.game_over = False
        return self._get_state()
    
    def _new_food(self):
        while True:
            x = random.randint(0, (self.width-self.grid_size)//self.grid_size) * self.grid_size
            y = random.randint(0, (self.height-self.grid_size)//self.grid_size) * self.grid_size
            if (x, y) not in self.snake:
                return (x, y)

    def _check_collision(self, x, y):
        return 1.0 if (x < 0 or x >= self.width or 
                      y < 0 or y >= self.height or 
                      (x, y) in self.snake) else 0.0
    
    def step(self, action):
        self.steps += 1
        self.steps_without_food += 1
        
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
            new_head[0] < 0 or 
            new_head[0] >= self.width or 
            new_head[1] < 0 or 
            new_head[1] >= self.height):
            self.game_over = True
            return self._get_state(), -1
        
        self.snake.insert(0, new_head)
        
        reward = 0.1
        
        if self.snake[0] == self.food:
            self.score += 1
            self.food = self._new_food()
            self.steps_without_food = 0
            reward = 1.0 + (len(self.snake) * 0.1)
        else:
            self.snake.pop()
        if self.steps_without_food >= self.max_steps_without_food:
            self.game_over = True
            reward = -0.5 
            
        return self._get_state(), reward
    
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
        
        state = [
            (food_x - head_x) / self.width,
            (food_y - head_y) / self.height,
            food_distance / (self.width**2 + self.height**2)**0.5,
            self.dx / self.grid_size,
            self.dy / self.grid_size,
            *food_direction,
            *danger_map,
            len(self.snake) / (self.width * self.height / (self.grid_size * self.grid_size)),
            self.steps / self.max_steps_without_food
        ]
        return np.array(state, dtype=np.float32)

def evaluate_network(network: NeuralNetwork, env_params: dict, games_per_network: int = 5) -> float:
    env = SnakeEnv(**env_params)
    total_score = 0
    total_steps = 0
    total_reward = 0
    
    for _ in range(games_per_network):
        state = env.reset()
        episode_reward = 0
        
        while not env.game_over:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_probs = network(state_tensor)
            action = action_probs.argmax().item()
            
            state, reward = env.step(action)
            episode_reward += reward
        
        total_score += env.score
        total_steps += env.steps
        total_reward += episode_reward
    avg_score = total_score / games_per_network
    avg_steps = total_steps / games_per_network
    avg_reward = total_reward / games_per_network
    
    fitness = (avg_score * 10.0 +
              avg_steps / 100.0 +
              avg_reward)
    
    return fitness

class ImprovedParallelGeneticAlgorithm:
    def __init__(self, 
                 population_size: int = 200,
                 mutation_rate: float = 0.3,
                 mutation_strength: float = 0.5,
                 elite_size: int = 10,
                 games_per_network: int = 5,
                 num_workers: int = None,
                 diversity_weight: float = 0.3, 
                 tournament_size: int = 3): 
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.elite_size = elite_size
        self.games_per_network = games_per_network
        self.num_workers = num_workers or mp.cpu_count()
        self.diversity_weight = diversity_weight
        self.tournament_size = tournament_size
        self.prev_avg_fitness = float('-inf')
        
        self.population = []
        self.fitness_scores = []
        for _ in range(population_size):
            network = NeuralNetwork()
            self.population.append(network)
    
    def calculate_diversity(self, network1: NeuralNetwork, network2: NeuralNetwork) -> float:
        weights1 = network1.get_weights()
        weights2 = network2.get_weights()
        total_diff = 0
        total_weights = 0
        
        for w1, w2 in zip(weights1, weights2):
            diff = np.sum(np.abs(w1 - w2))
            total_diff += diff
            total_weights += w1.size
        
        return total_diff / total_weights
    
    def get_population_diversity(self, network: NeuralNetwork) -> float:
        diversities = []
        sample_size = min(20, len(self.population))
        sample_population = random.sample(self.population, sample_size)
        
        for other in sample_population:
            if other != network:
                div = self.calculate_diversity(network, other)
                diversities.append(div)
        
        return np.mean(diversities) if diversities else 0
    
    def evaluate_fitness_parallel(self, env_params: dict) -> List[float]:
        # 使用更小的批次大小来处理评估
        batch_size = self.population_size // (self.num_workers * 2)
        batches = [self.population[i:i + batch_size] for i in range(0, len(self.population), batch_size)]
        
        all_base_fitness_scores = []
        with mp.Pool(self.num_workers) as pool:
            for batch in batches:
                eval_func = partial(evaluate_network, 
                                  env_params=env_params,
                                  games_per_network=self.games_per_network)
                
                batch_scores = pool.map(eval_func, batch)
                all_base_fitness_scores.extend(batch_scores)
                
            # 计算多样性分数
            diversity_scores = []
            for network in self.population:
                diversity = self.get_population_diversity(network)
                diversity_scores.append(diversity)
            
            if diversity_scores:
                min_div = min(diversity_scores)
                max_div = max(diversity_scores)
                if max_div > min_div:
                    diversity_scores = [(d - min_div) / (max_div - min_div) for d in diversity_scores]
                else:
                    diversity_scores = [1.0] * len(diversity_scores)
            
            self.fitness_scores = [
                base_fit * (1 - self.diversity_weight) + div * self.diversity_weight 
                for base_fit, div in zip(all_base_fitness_scores, diversity_scores)
            ]
            
            return self.fitness_scores
    
    def select_parent(self) -> NeuralNetwork:
        tournament = random.sample(list(enumerate(self.fitness_scores)), self.tournament_size)
        winner_idx = max(tournament, key=lambda x: x[1])[0]
        return self.population[winner_idx]
    
    def adaptive_mutation(self, avg_fitness: float):
        if avg_fitness < self.prev_avg_fitness:
            self.mutation_strength = max(0.1, self.mutation_strength * 0.9)
            self.mutation_rate = min(0.8, self.mutation_rate * 1.1)
        else:
            self.mutation_strength = min(0.5, self.mutation_strength * 1.05)
            self.mutation_rate = max(0.3, self.mutation_rate * 0.95)
        
        self.prev_avg_fitness = avg_fitness
    
    def crossover(self, parent1: NeuralNetwork, parent2: NeuralNetwork) -> NeuralNetwork:
        child = NeuralNetwork()
        parent1_weights = parent1.get_weights()
        parent2_weights = parent2.get_weights()
        child_weights = []
        
        for w1, w2 in zip(parent1_weights, parent2_weights):
            alpha = np.random.rand(*w1.shape)
            child_weight = alpha * w1 + (1 - alpha) * w2
            child_weights.append(child_weight)
        
        child.set_weights(child_weights)
        return child

    def mutate(self, network: NeuralNetwork):
        weights = network.get_weights()
        mutated_weights = []
        
        for weight in weights:
            if random.random() < self.mutation_rate:
                mutation = np.random.normal(0, self.mutation_strength, weight.shape)
                mutation_mask = np.random.rand(*weight.shape) < 0.3
                mutation = mutation * mutation_mask
                mutated_weight = weight + mutation
                mutated_weights.append(mutated_weight)
            else:
                mutated_weights.append(weight)
        
        network.set_weights(mutated_weights)
    
    def evolve(self):
        avg_fitness = sum(self.fitness_scores) / len(self.fitness_scores)
        self.adaptive_mutation(avg_fitness)

        sorted_indices = np.argsort(self.fitness_scores)[::-1]
        new_population = [self.population[i] for i in sorted_indices[:self.elite_size]]
        while len(new_population) < self.population_size:
            parent1 = self.select_parent()
            parent2 = self.select_parent()
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            new_population.append(child)
        
        self.population = new_population


def train_snake_ga_parallel_improved(generations=5000):
    # 设置进程启动方法
    if sys.platform != 'darwin':  # 如果不是 MacOS
        mp.set_start_method('fork', force=True)
    
    # 增加系统文件描述符限制
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    except Exception as e:
        print(f"Warning: Could not set file descriptor limit: {e}")
    
    env_params = {
        'width': WIDTH,
        'height': HEIGHT,
        'grid_size': GRID_SIZE,
        'max_steps_without_food': 200
    }
    
    # 减小种群大小和每个网络的游戏次数以加快评估
    ga = ImprovedParallelGeneticAlgorithm(
        population_size=100,  # 减小种群大小
        games_per_network=3,  # 减少每个网络的评估次数
        num_workers=min(os.cpu_count(), 4)  # 限制CPU核心数
    )
    
    best_score = 0
    best_networks = []
    
    try:
        for generation in range(generations):
            fitness_scores = ga.evaluate_fitness_parallel(env_params)
            max_fitness = max(fitness_scores)
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            best_idx = fitness_scores.index(max_fitness)
            
            # 每10代才进行完整的测试评估
            if generation % 10 == 0:
                env = SnakeEnv(**env_params)
                test_scores = []
                num_test_games = 5  # 减少测试游戏次数
                
                for _ in range(num_test_games):
                    state = env.reset()
                    while not env.game_over:
                        state_tensor = torch.FloatTensor(state).unsqueeze(0)
                        with torch.no_grad():
                            action = ga.population[best_idx](state_tensor).argmax().item()
                        state, _ = env.step(action)
                    test_scores.append(env.score)
                
                current_score = sum(test_scores) / len(test_scores)
                current_max = max(test_scores)
                current_min = min(test_scores)
                
                if current_score > best_score:
                    best_score = current_score
                    best_network = copy.deepcopy(ga.population[best_idx])
                    best_networks.append((best_score, best_network))
                    best_networks.sort(key=lambda x: x[0], reverse=True)
                    best_networks = best_networks[:5]
                    torch.save(best_network.state_dict(), f"snake_ga_best_{best_score:.1f}.pth")
                
                print(f"Generation {generation + 1}/{generations}")
                print(f"Best Score: {current_score:.1f} (min: {current_min}, max: {current_max})")
                print(f"Average Fitness: {avg_fitness:.2f}")
                print(f"All-time Best: {best_score:.1f}")
                print(f"Mutation Rate: {ga.mutation_rate:.2f}, Strength: {ga.mutation_strength:.2f}")
                print(f"Using {ga.num_workers} CPU cores")
                print("------------------------")
            else:
                # 简单的进度打印
                if generation % 5 == 0:  # 每5代打印一次简单进度
                    print(f"Generation {generation + 1}: Avg Fitness = {avg_fitness:.2f}")
            
            ga.evolve()
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving best network...")
        if best_networks:
            torch.save(best_networks[0][1].state_dict(), "snake_ga_interrupted.pth")
    
    return best_networks[0][1] if best_networks else None

if __name__ == "__main__":
    best_network = train_snake_ga_parallel_improved()
