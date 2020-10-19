import numpy as np

class AckleyES(object):
    def __init__(self, crossover_prob=0.9, mutation_prob=0.4, pop_size=100):
        self.p_c = crossover_prob
        self.p_m = mutation_prob
        self.population_size = pop_size
        self.population = None
        self.mutation_steps = None
        self.pop_fitness = None
        self.num_fitness_eval = 0
        self.n = 30
    
    def run(self, parameter_list):
        """
        docstring
        """
        pass

    def random_init_population(self):
        """
        Randomly initialize the population.
        """
        rng = np.random.default_rng()
        self.population = rng.uniform(-15, 15, (self.population_size, self.n))
        self.mutation_steps = rng.random((self.population_size, self.n))
        self.pop_fitness = np.array([self.fitness(i) for i in self.population])
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

    def select_survivors(self):
        """
        Deterministic and elitist survivor selection by
        replacing individuals by the offspring.
        """
        pass

    def select_parents(self):
        """
        Select the parents to generate the next offspring.
        """
        pass