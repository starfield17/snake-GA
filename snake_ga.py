import numpy as np
import random
from typing import List
import torch
import torch.nn as nn
import threading
from queue import Queue
import copy
import time

class NetworkEvaluator(threading.Thread):
    def __init__(self, task_queue, result_queue, env_params):
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.env_params = env_params
        self.daemon = True
    
    def run(self):
        while True:
            try:
                idx, network = self.task_queue.get(timeout=1)
                if network is None:
                    break
                
                fitness = self.evaluate_single_network(network)
                self.result_queue.put((idx, fitness))
                self.task_queue.task_done()
            except:
                break
    
    def evaluate_single_network(self, network, games=3):
        env = SnakeEnv(**self.env_params)
        total_score = 0
        total_steps = 0
        total_reward = 0
        
        for _ in range(games):
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
        
        avg_score = total_score / games
        avg_steps = total_steps / games
        avg_reward = total_reward / games
        
        fitness = (avg_score * 10.0 + avg_steps / 100.0 + avg_reward)
        return fitness

class SimpleGeneticAlgorithm:
    def __init__(self, 
                 population_size: int = 100,
                 mutation_rate: float = 0.3,
                 mutation_strength: float = 0.5,
                 elite_size: int = 5,
                 num_threads: int = 4):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.elite_size = elite_size
        self.num_threads = num_threads
        
        self.population = []
        self.fitness_scores = []
        for _ in range(population_size):
            network = NeuralNetwork()
            self.population.append(network)
    
    def evaluate_population(self, env_params: dict) -> List[float]:
        task_queue = Queue()
        result_queue = Queue()
        
        # 创建任务
        for idx, network in enumerate(self.population):
            task_queue.put((idx, network))
        
        # 启动线程
        threads = []
        for _ in range(self.num_threads):
            thread = NetworkEvaluator(task_queue, result_queue, env_params)
            thread.start()
            threads.append(thread)
        
        # 等待所有任务完成
        task_queue.join()
        
        # 停止线程
        for _ in range(self.num_threads):
            task_queue.put((None, None))
        for thread in threads:
            thread.join()
        
        # 收集结果
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        
        # 按原始索引排序结果
        results.sort(key=lambda x: x[0])
        self.fitness_scores = [r[1] for r in results]
        return self.fitness_scores
    
    def select_parent(self) -> NeuralNetwork:
        tournament_size = 3
        tournament = random.sample(list(enumerate(self.fitness_scores)), tournament_size)
        winner_idx = max(tournament, key=lambda x: x[1])[0]
        return self.population[winner_idx]
    
    def crossover(self, parent1: NeuralNetwork, parent2: NeuralNetwork) -> NeuralNetwork:
        child = NeuralNetwork()
        parent1_weights = parent1.get_weights()
        parent2_weights = parent2.get_weights()
        child_weights = []
        
        for w1, w2 in zip(parent1_weights, parent2_weights):
            mask = np.random.rand(*w1.shape) < 0.5
            child_weight = np.where(mask, w1, w2)
            child_weights.append(child_weight)
        
        child.set_weights(child_weights)
        return child
    
    def mutate(self, network: NeuralNetwork):
        weights = network.get_weights()
        mutated_weights = []
        
        for weight in weights:
            if random.random() < self.mutation_rate:
                mutation = np.random.normal(0, self.mutation_strength, weight.shape)
                mutated_weight = weight + mutation
                mutated_weights.append(mutated_weight)
            else:
                mutated_weights.append(weight)
        
        network.set_weights(mutated_weights)
    
    def evolve(self):
        sorted_indices = np.argsort(self.fitness_scores)[::-1]
        new_population = [copy.deepcopy(self.population[i]) for i in sorted_indices[:self.elite_size]]
        
        while len(new_population) < self.population_size:
            parent1 = self.select_parent()
            parent2 = self.select_parent()
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            new_population.append(child)
        
        self.population = new_population

def train_snake(generations=5000):
    env_params = {
        'width': WIDTH,
        'height': HEIGHT,
        'grid_size': GRID_SIZE,
        'max_steps_without_food': 200
    }
    
    ga = SimpleGeneticAlgorithm()
    best_score = 0
    best_networks = []
    start_time = time.time()
    
    try:
        for generation in range(generations):
            # 评估当前种群
            fitness_scores = ga.evaluate_population(env_params)
            max_fitness = max(fitness_scores)
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            best_idx = fitness_scores.index(max_fitness)
            
            # 测试最佳网络
            if generation % 10 == 0:
                env = SnakeEnv(**env_params)
                test_scores = []
                num_test_games = 5
                
                for _ in range(num_test_games):
                    state = env.reset()
                    while not env.game_over:
                        state_tensor = torch.FloatTensor(state).unsqueeze(0)
                        with torch.no_grad():
                            action = ga.population[best_idx](state_tensor).argmax().item()
                        state, _ = env.step(action)
                    test_scores.append(env.score)
                
                current_score = sum(test_scores) / len(test_scores)
                if current_score > best_score:
                    best_score = current_score
                    best_network = copy.deepcopy(ga.population[best_idx])
                    best_networks.append((best_score, best_network))
                    best_networks.sort(key=lambda x: x[0], reverse=True)
                    best_networks = best_networks[:5]
                    torch.save(best_network.state_dict(), f"snake_best_{best_score:.1f}.pth")
                
                elapsed_time = time.time() - start_time
                print(f"\nGeneration {generation + 1}/{generations}")
                print(f"Time elapsed: {elapsed_time:.0f}s")
                print(f"Best Score: {current_score:.1f} (max: {max(test_scores)})")
                print(f"Average Fitness: {avg_fitness:.2f}")
                print(f"All-time Best: {best_score:.1f}")
                print("------------------------")
            else:
                print(f"Generation {generation + 1}: Avg Fitness = {avg_fitness:.2f}", end='\r')
            
            ga.evolve()
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving best network...")
        if best_networks:
            torch.save(best_networks[0][1].state_dict(), "snake_interrupted.pth")
    
    return best_networks[0][1] if best_networks else None

if __name__ == "__main__":
    best_network = train_snake()
