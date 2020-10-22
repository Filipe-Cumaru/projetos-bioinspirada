import numpy as np

class AckleyES(object):
    def __init__(self, crossover_prob=1.0, mutation_prob=1.0, pop_size=30, offspring_size=200):
        self.p_c = crossover_prob
        self.p_m = mutation_prob
        self.population_size = pop_size
        self.offspring_size = offspring_size
        self.population = None
        self.pop_fitness = None
        self.num_fitness_eval = 0
        self.n = 30
        # The learning rates.
        self.global_lr = 1 / (2*self.n)**0.5
        self.local_lr = 1 / (2*(self.n)**0.5)**0.5
    
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
        self.population = list(zip(solutions, mutation_steps))
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
        Generate an offspring from the current population.
        It first generates the candidate solutions then 
        the mutation parameters.
        """
        candidates = self.recombine_solutions()
        mutation_parameters = self.recombine_parameters()
        offspring = list(zip(candidates, mutation_parameters))
        return offspring
    
    def recombine_solutions(self):
        """
        Recombination of the solution vector of an individual.
        We use a global discrete recombination scheme.
        """
        rng = np.random.default_rng()
        pop_solutions = np.array([sol for sol, _ in self.population])
        offspring_solutions = []
        cols = np.arange(self.n)
        indices = np.arange(self.population_size)
        for _ in range(self.offspring_size):
            rows = rng.choice(indices, size=self.n, replace=False)
            new_solution = pop_solutions[rows, cols]
            offspring_solutions.append(new_solution)
        
        return offspring_solutions
    
    def recombine_parameters(self):
        """
        Recombination of the mutation parameters of an individual.
        We use a local whole arithmetical recombination scheme.
        """
        rng = np.random.default_rng()
        alpha = 0.6
        pop_parameters = np.array([param for _, param in self.population])
        offspring_parameters = []
        for _ in range(int(self.offspring_size / 2)):
            rows = rng.choice(np.arange(self.population_size), size=2, replace=False)
            p1, p2 = pop_parameters[rows]
            x1, x2 = alpha*p1 + (1 - alpha)*p2, alpha*p2 + (1 - alpha)*p1
            offspring_parameters.extend((x1, x2))
        
        return offspring_parameters

    def mutate(self, offspring):
        """
        Introduce changes to the offspring. First we mutate 
        the parameters then we use them to mutate the solutions.
        """
        solutions = np.array([sol for sol, _ in offspring])
        parameters = np.array([param for _, param in offspring])
        mutated_parameters = self.mutate_parameters(parameters)
        mutated_solutions = self.mutate_solutions(mutated_parameters, solutions)
        mutated_offspring = list(zip(mutated_solutions, mutated_parameters))
        return mutated_offspring
    
    def mutate_parameters(self, parameters):
        """
        Apply a non-correlated mutation with different
        parameters for each variable.
        """
        rng = np.random.default_rng()
        eps = 1e-3

        # Compute the mutation.
        gaussian_var = rng.standard_normal(1)[0]
        gaussian_vector = rng.standard_normal(self.n)
        A = self.global_lr*gaussian_var
        B = self.local_lr*gaussian_vector
        mutated_parameters = parameters*np.exp(A + B)

        # Check if the parameters aren't too small.
        small_params = mutated_parameters <= eps
        mutated_parameters[small_params] = eps

        return mutated_parameters

    def mutate_solutions(self, parameters, solutions):
        """
        Apply the mutation to the candidate solutions.
        """
        rng = np.random.default_rng()
        gaussian_vector = rng.standard_normal(self.n)
        mutated_solutions = solutions + parameters*gaussian_vector
        return mutated_solutions

    def select_survivors(self, offspring):
        """
        Deterministic and elitist survivor selection by
        replacing the current individuals by the µ best
        individuals from the offspring, a.k.a, a (µ,λ) scheme.
        """
        # Compute the fitness for each child and select the best ones.
        offspring_fitness = np.array([self.fitness(i) for i, _ in self.population])
        best_offspring_indices = np.argsort(offspring_fitness)[:self.population_size]
        self.population = offspring[best_offspring_indices]

    def select_parents(self):
        """
        Select a pair of parents for local recombination.
        """
        rng = np.random.default_rng()
        parents = rng.choice(self.population, 2, replace=False)
        return parents