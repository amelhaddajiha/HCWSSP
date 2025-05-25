import numpy as np
import random
import matplotlib.pyplot as plt

# Configuration du problème
n_jobs = 20
n_stages = 5
max_skilled_welders = [3, 3, 3, 3, 3]
max_ordinary_welders = [5, 5, 5, 5, 5]
max_robots = [4, 4, 4, 4, 4]

processing_times = {(0,0):(8,4),(0,1):(16,8),(0,2):(16,12),(0,3):(10,6),(0,4):(12,5),
                    (1,0):(10,0),(1,1):(8,0),(1,2):(6,0),(1,3):(6,0),(1,4):(9,0),
                    (2,0):(12,0),(2,1):(12,0),(2,2):(8,0),(2,3):(8,0),(2,4):(7,0),
                    (3,0):(9,3),(3,1):(11,6),(3,2):(14,9),(3,3):(7,4),(3,4):(10,3),
                    (4,0):(15,0),(4,1):(10,0),(4,2):(9,0),(4,3):(12,0),(4,4):(6,0),
                    (5,0):(7,2),(5,1):(13,7),(5,2):(11,8),(5,3):(9,5),(5,4):(13,4),
                    (6,0):(11,0),(6,1):(9,0),(6,2):(7,0),(6,3):(10,0),(6,4):(8,0),
                    (7,0):(14,4),(7,1):(12,6),(7,2):(10,7),(7,3):(11,3),(7,4):(15,6),
                    (8,0):(6,0),(8,1):(7,0),(8,2):(9,0),(8,3):(13,0),(8,4):(11,0),
                    (9,0):(10,3),(9,1):(15,8),(9,2):(12,9),(9,3):(8,4),(9,4):(14,5),
                    (10,0):(13,0),(10,1):(11,0),(10,2):(10,0),(10,3):(7,0),(10,4):(9,0),
                    (11,0):(8,2),(11,1):(14,7),(11,2):(13,8),(11,3):(10,5),(11,4):(12,4),
                    (12,0):(9,0),(12,1):(8,0),(12,2):(11,0),(12,3):(12,0),(12,4):(7,0),
                    (13,0):(12,3),(13,1):(10,6),(13,2):(15,9),(13,3):(9,4),(13,4):(11,3),
                    (14,0):(11,0),(14,1):(13,0),(14,2):(12,0),(14,3):(6,0),(14,4):(10,0),
                    (15,0):(7,2),(15,1):(9,7),(15,2):(14,8),(15,3):(12,5),(15,4):(8,4),
                    (16,0):(10,0),(16,1):(6,0),(16,2):(8,0),(16,3):(11,0),(16,4):(13,0),
                    (17,0):(13,3),(17,1):(11,6),(17,2):(9,9),(17,3):(14,4),(17,4):(7,3),
                    (18,0):(8,0),(18,1):(10,0),(18,2):(7,0),(18,3):(15,0),(18,4):(12,0),
                    (19,0):(11,2),(19,1):(12,7),(19,2):(13,8),(19,3):(6,5),(19,4):(9,4)}

P_run = [1.0] * n_stages
P_sb = [0.5] * n_stages

class Solution:
    def __init__(self):
        self.permutation = list(range(n_jobs))
        self.resources = []
        self.objectives = [np.inf, np.inf]
        
    def copy(self):
        new_sol = Solution()
        new_sol.permutation = self.permutation.copy()
        new_sol.resources = [[r.copy() for r in stage] for stage in self.resources]
        new_sol.objectives = self.objectives.copy()
        return new_sol

class MOEADSolver:
    def __init__(self, pop_size=100, max_gen=200, crossover_rate=0.9, mutation_rate=0.1, neighborhood=20):
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.neighborhood_size = neighborhood
        self.weights = []
        self.neighbors = []
        self.population = []
        self.ideal_point = []
        self.hv_log = []
        self.gd_log = []

    def generate_weights(self):
        for i in range(self.pop_size):
            w = i / (self.pop_size - 1)
            self.weights.append(np.array([w, 1 - w]))

    def initialize_neighbors(self):
        self.neighbors = []
        for i in range(len(self.weights)):
            distances = []
            for j in range(len(self.weights)):
                distances.append(np.linalg.norm(self.weights[i] - self.weights[j]))
            indices = np.argsort(distances)
            # Conversion en liste Python
            self.neighbors.append(indices[:self.neighborhood_size].tolist())

    def initialize_population(self):
        self.generate_weights()
        self.initialize_neighbors()
        self.population = [self.create_random_solution() for _ in range(self.pop_size)]
        self.update_ideal_point()

    def create_random_solution(self):
        sol = Solution()
        random.shuffle(sol.permutation)
        sol.resources = self.generate_resources()
        self.evaluate(sol)
        return sol

    def generate_resources(self):
        resources = []
        for stage in range(n_stages):
            stage_res = []
            for _ in range(n_jobs):
                if stage == 0:
                    N_s = random.randint(1, max_skilled_welders[stage])
                    res = [N_s, 0, 0]
                else:
                    if random.random() < 0.5:
                        N_o = random.randint(1, max_ordinary_welders[stage])
                        res = [0, N_o, 0]
                    else:
                        R = random.randint(1, max_robots[stage])
                        res = [0, 0, R]
                stage_res.append(res)
            resources.append(stage_res)
        return resources

    def evaluate(self, sol):
        job_times = np.zeros((n_stages, n_jobs))
        EC_r = 0
        EC_s = 0

        for stage in range(n_stages):
            stage_start = 0
            stage_total = 0
            
            for idx, job in enumerate(sol.permutation):
                N_s, N_o, R = sol.resources[stage][job]
                
                if stage == 0:
                    N_s = max(1, N_s)
                else:
                    if N_o + R == 0:
                        if random.random() < 0.5:
                            N_o = 1
                        else:
                            R = 1
                
                if stage == 0:
                    p = processing_times[(job, stage)][1]
                    duration = p / N_s
                else:
                    p = processing_times[(job, stage)][0]
                    workers = N_o if N_o > 0 else R
                    duration = p / max(workers, 1)
                
                prev_time = job_times[stage-1][idx] if stage > 0 else 0
                start = max(stage_start, prev_time)
                end = start + duration
                
                job_times[stage][idx] = end
                stage_start = end
                stage_total += duration
                
                EC_r += (N_s + N_o + R) * duration * P_run[stage]

            EC_s += (stage_start - stage_total) * P_sb[stage]

        sol.objectives = [job_times[-1][-1], EC_r + EC_s]

    def tchebycheff(self, sol, weight):
        return max(weight[i] * abs(sol.objectives[i] - self.ideal_point[i]) for i in range(2))

    def update_ideal_point(self):
        self.ideal_point = [
            min(sol.objectives[0] for sol in self.population),
            min(sol.objectives[1] for sol in self.population)
        ]

    def solve(self):
        self.initialize_population()
        
        for gen in range(self.max_gen):
            for i in range(self.pop_size):
                if len(self.neighbors[i]) >= 2:  # Vérification de la taille du voisinage
                    # Sélection des parents dans le voisinage
                    k, l = random.sample(self.neighbors[i], 2)
                    parent1 = self.population[k]
                    parent2 = self.population[l]
                    
                    # Reproduction
                    child, _ = self.crossover(parent1, parent2)
                    self.mutate(child)
                    self.evaluate(child)
                    
                    # Mise à jour du point idéal
                    self.ideal_point = [
                        min(self.ideal_point[0], child.objectives[0]),
                        min(self.ideal_point[1], child.objectives[1])
                    ]
                    
                    # Mise à jour des solutions voisines
                    for j in self.neighbors[i]:
                        if self.tchebycheff(child, self.weights[j]) < self.tchebycheff(self.population[j], self.weights[j]):
                            self.population[j] = child.copy()
            
            # Logging
            if gen % 10 == 0:
                self.hv_log.append(self.calc_hypervolume())
                self.gd_log.append(self.calc_generational_distance())
                print(f"Gen {gen}: HV={self.hv_log[-1]:.1f}, GD={self.gd_log[-1]:.2f}")
        
        return self.population

    def crossover(self, parent1, parent2):
        child1, child2 = parent1.copy(), parent2.copy()
        if random.random() < self.crossover_rate:
            pt = random.randint(1, len(parent1.permutation)-1)
            child1.permutation = parent1.permutation[:pt] + [j for j in parent2.permutation if j not in parent1.permutation[:pt]]
            child2.permutation = parent2.permutation[:pt] + [j for j in parent1.permutation if j not in parent2.permutation[:pt]]
        return child1, child2

    def mutate(self, sol):
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(n_jobs), 2)
            sol.permutation[i], sol.permutation[j] = sol.permutation[j], sol.permutation[i]
        for stage in range(n_stages):
            for job in range(n_jobs):
                if random.random() < self.mutation_rate/10:
                    if stage == 0:
                        sol.resources[stage][job][0] = random.randint(1, max_skilled_welders[stage])
                    else:
                        if random.random() < 0.5:
                            sol.resources[stage][job][1] = random.randint(1, max_ordinary_welders[stage])
                            sol.resources[stage][job][2] = 0
                        else:
                            sol.resources[stage][job][2] = random.randint(1, max_robots[stage])
                            sol.resources[stage][job][1] = 0

    def calc_hypervolume(self):
        if not self.population:
            return 0
        ref = [max(sol.objectives[0] for sol in self.population) * 1.1,
               max(sol.objectives[1] for sol in self.population) * 1.1]
        sorted_pop = sorted(self.population, key=lambda x: x.objectives[0])
        hv = 0
        prev = [0, ref[1]]
        for sol in sorted_pop:
            width = sol.objectives[0] - prev[0]
            height = ref[1] - sol.objectives[1]
            hv += width * height
            prev = sol.objectives
        return hv

    def calc_generational_distance(self):
        reference_front = np.array([[450, 2500], [500, 2200], [550, 2000]])
        distances = []
        for sol in self.population:
            min_dist = min(np.linalg.norm(sol.objectives - ref) for ref in reference_front)
            distances.append(min_dist)
        return np.mean(distances)

    def plot_results(self):
        plt.figure(figsize=(15,5))
        plt.subplot(131)
        plt.scatter([s.objectives[1] for s in self.population], 
                    [s.objectives[0] for s in self.population])
        plt.xlabel('TEC'), plt.ylabel('Makespan')
        plt.subplot(132)
        plt.plot(self.hv_log, marker='o')
        plt.title('Hypervolume Evolution')
        plt.subplot(133)
        plt.plot(self.gd_log, marker='s', color='red')
        plt.title('Generational Distance')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    solver = MOEADSolver(
        pop_size=100,
        max_gen=200,
        crossover_rate=0.85,
        mutation_rate=0.15,
        neighborhood=20
    )
    population = solver.solve()
    solver.plot_results()
    
    print("Top 5 solutions:")
    # Filtrage des solutions non dominées
    non_dominated = []
    for sol in population:
        is_dominated = False
        for other in population:
            if (other.objectives[0] <= sol.objectives[0] and 
                other.objectives[1] <= sol.objectives[1] and 
                (other.objectives[0] < sol.objectives[0] or other.objectives[1] < sol.objectives[1])):
                is_dominated = True
                break
        if not is_dominated:
            non_dominated.append(sol)
    
    non_dominated.sort(key=lambda x: x.objectives[0])
    for sol in non_dominated[:5]:
        print(f"Makespan: {sol.objectives[0]:.1f}, TEC: {sol.objectives[1]:.1f}")
        print(f"Sequence: {sol.permutation}")
        print(f"Resources (Stage 0): {sol.resources[0][:3]}...\n")