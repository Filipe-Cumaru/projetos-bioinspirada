import numpy as np

class AckleyES(object):
    def __init__(self, crossover_prob=0.9, mutation_prob=1.0, pop_size=30, offspring_size=200):
        self.p_c = crossover_prob
        self.p_m = mutation_prob
        self.population_size = pop_size
        self.offspring_size = offspring_size
        self.population = None
        self.pop_fitness = None
        self.num_fitness_eval = 0
        self.n = 30
    
    def run(self, parameter_list):
        """
        Main function. Runs the evolutionary strategy algorithm.
        """
        pass

    def random_init_population(self):
        """
        Randomly initialize the population.
        """
        rng = np.random.default_rng()
        solutions = rng.uniform(-15, 15, (self.population_size, self.n))
        mutation_steps = rng.random((self.population_size, self.n))
        # Each individual is a tuple containing a candidate solution
        # and its mutation_steps.
        self.population = zip(solutions, mutation_steps)
        self.pop_fitness = np.array([self.fitness(i) for i, _ in self.population])
        self.num_fitness_eval = 0
    
    def fitness(self, candidate):
        """
        Fitness function.
        """
        solution = np.zeros(self.n)
        error = np.linalg.norm(candidate - solution)
        return 1 / (error + 1)

    def recombine(self):
        """
        docstring
        """
        pass

    def mutate(self):
        """
        docstring
        """
        pass

    def select_survivors(self, offspring):
        """
        Deterministic and elitist survivor selection by
        replacing the current individuals by the µ best
        individuals from the offspring, a.k.a, a (µ,λ) scheme.
        """
        # Compute the fitness for each child and select the best ones.
        offspring_fitness = np.array([self.fitness(i) for i, _ in self.population])
        best_offspring_indexes = np.argsort(offspring_fitness)[:self.population_size]
        self.population = offspring[best_offspring_indexes]

    def select_parents(self):
        """
        Select a pair of parents for local recombination.
        """
        rng = np.random.default_rng()
        parents = rng.choice(self.population, 2, replace=False)
        return parents