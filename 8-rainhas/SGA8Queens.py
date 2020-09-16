import numpy as np

class SGA8Queens(object):
    def __init__(self, crossover_prob=0.9, mutation_prob=0.4, pop_size=100):
        # Crossover probability.
        self.p_c = crossover_prob

        # Mutation probability.
        self.p_m = mutation_prob

        # Population size.
        self.pop_size = pop_size
        self.population = None
        self.pop_fitness = None

        # Number of evaluations of the fitness function.
        self.num_fitness_eval = 0

    def fitness(self, table):
        # Based on https://bit.ly/32zycds.
        eps = 0.01
        N = 8
        # Number of conflicts.
        result = 0
        # Occurences of a queen in a row.
        f_row = np.zeros(8)
        # Occurences of a queen in the main diagonal.
        f_mdiag = np.zeros(16)
        # Occurences of a queen in the secondary diagonal.
        f_sdiag = np.zeros(16)

        for i in range(N):
            f_row[table[i] - 1] += 1
            f_mdiag[table[i] + i - 1] += 1
            f_sdiag[N - table[i] + i - 1] += 1
        
        for i in range(2*N):
            x, y, z = 0, 0, 0
            if i < N:
                x = f_row[i]
            y = f_mdiag[i]
            z = f_sdiag[i]
            result += ((x * (x - 1)) + (y * (y - 1)) + (z * (z - 1))) / 2
        
        self.num_fitness_eval += 1
        
        return 1 / (result + eps)

    def select_parents(self):
        rng = np.random.default_rng()
        candidates = rng.choice(self.population, size=5)
        candidates_fitness = np.array([self.fitness(x) for x in candidates])
        # Inside the brackets: find the index of the two highest fitness.
        p1, p2 = candidates[np.argsort(-candidates_fitness)[:2]]
        return p1, p2

    def crossover(self, p1, p2):
        rng = np.random.default_rng()
        c1 = np.zeros(8, dtype="int64")
        c2 = np.zeros(8, dtype="int64")

        if rng.uniform() < self.p_c:
            i = rng.choice(8)
            c1[:i], c2[:i] = p1[:i], p2[:i]
            it1, it2 = i, i
            for j in range(8):
                if p2[j] not in c1:
                    c1[it1] = p2[j]
                    it1 += 1
                if p1[j] not in c2:
                    c2[it2] = p1[j]
                    it2 += 1
        else:
            c1, c2 = p1[:], p2[:]
        
        return c1, c2

    def mutate(self, child):
        rng = np.random.default_rng()

        if rng.uniform() < self.p_m:
            i, j = rng.choice(8, size=2, replace=False)
            child[i], child[j] = child[j], child[i]
        
        return child

    def update_population(self, c1, c2):
        weakest_solutions = np.argsort(self.pop_fitness)[:2]
        self.population[weakest_solutions[0]] = c1
        self.population[weakest_solutions[1]] = c2
        self.pop_fitness[weakest_solutions[0]] = self.fitness(c1)
        self.pop_fitness[weakest_solutions[1]] = self.fitness(c2)

    def run(self):
        rng = np.random.default_rng()
        self.population = [rng.permutation(np.arange(1, 9)) for i in range(self.pop_size)]
        self.pop_fitness = np.empty(self.pop_size)

        while self.num_fitness_eval < 10000 and self.pop_fitness.max() != 100:
            p1, p2 = self.select_parents()
            c1, c2 = self.crossover(p1, p2)
            c1 = self.mutate(c1)
            c2 = self.mutate(c2)
            self.pop_fitness = np.array([self.fitness(x) for x in self.population])
            self.update_population(c1, c2)

