import numpy as np
from ackley import ackley_function

import matplotlib.pyplot as plt

class AckleyES(object):
    def __init__(self, pop_size=30, offspring_size=200, max_iter=200000, tol=-0.01):
        self.population_size = pop_size
        self.offspring_size = offspring_size
        self.population = None
        self.pop_fitness = None
        self.n = 30
        self.max_iter = max_iter
        self.tol = tol
        # The learning rates.
        self.global_lr = 1e-1 / (2*self.n)**0.5 # best 1e-1
        self.local_lr = 1e-3 / (2*(self.n)**0.5)**0.5 # best 1e-3
    
    def run(self):
        """
        Main function. Runs the evolutionary strategy algorithm.
        """
        self.random_init_population()
        num_iter = 0

        while self.pop_fitness.max() < self.tol and num_iter < self.max_iter:
            print("[{}] Starting...".format(num_iter))
            offspring = self.recombine()
            mutated_offspring = self.mutate(offspring)
            self.select_survivors(mutated_offspring)
            print("[{}] Done!".format(num_iter))
            num_iter += 1

        index = self.pop_fitness.argmax()
        report = {
            "convergence": num_iter < 100000,
            "iterations": num_iter,
            "min error function": -self.pop_fitness.max(),
            "min error solution": np.linalg.norm(self.population[index, 0])
        }

        return report

    def random_init_population(self):
        """
        Randomly initialize the population.
        """
        rng = np.random.default_rng()
        solutions = rng.uniform(-15, 15, (self.population_size, self.n))
        mutation_steps = rng.uniform(0, 0.1, (self.population_size, self.n))
        # Each individual is a tuple containing a candidate solution
        # and its mutation_steps.
        self.population = np.array(list(zip(solutions, mutation_steps)))
        self.pop_fitness = np.array([self.fitness(i) for i, _ in self.population])
    
    def fitness(self, candidate):
        """
        Fitness function.
        """
        return -np.abs(ackley_function(candidate))

    def recombine(self):
        """
        Generate an offspring from the current population.
        It first generates the candidate solutions then 
        the mutation parameters.
        """
        candidates = self.recombine_solutions()
        mutation_parameters = self.recombine_parameters()
        offspring = np.array(list(zip(candidates, mutation_parameters)))
        return offspring
    
    def recombine_solutions(self):
        """
        Recombination of the solution vector of an individual.
        We use a global discrete recombination scheme.
        """
        rng = np.random.default_rng()
        pop_solutions = self.population[:, 0]
        cols = np.arange(self.n)
        indices = np.arange(self.population_size)
        offspring_solutions = [
            pop_solutions[rng.choice(indices, size=self.n, replace=False), cols]
            for _ in range(self.offspring_size)
        ]
        return np.array(offspring_solutions)
    
    def recombine_parameters(self):
        """
        Recombination of the mutation parameters of an individual.
        We use a local whole arithmetical recombination scheme.
        """
        rng = np.random.default_rng()
        pop_parameters = self.population[:, 1]
        offspring_parameters = []
        for _ in range(self.offspring_size):
            rows = rng.choice(np.arange(self.population_size), size=2, replace=False)
            p1, p2 = pop_parameters[rows]
            x = (p1 + p2) / 2
            offspring_parameters.append(x)
        
        return np.array(offspring_parameters)

    def mutate(self, offspring):
        """
        Introduce changes to the offspring. First we mutate 
        the parameters then we use them to mutate the solutions.
        """
        solutions = offspring[:, 0]
        parameters = offspring[:, 1]
        mutated_parameters = self.mutate_parameters(parameters)
        mutated_solutions = self.mutate_solutions(mutated_parameters, solutions)
        mutated_offspring = np.array(list(zip(mutated_solutions, mutated_parameters)))
        return mutated_offspring
    
    def mutate_parameters(self, parameters):
        """
        Apply a non-correlated mutation with different
        parameters for each variable.
        """
        rng = np.random.default_rng()
        eps = 1e-5

        # Compute the mutation.
        gaussian_var = rng.standard_normal(1)[0]
        gaussian_vector = rng.standard_normal((self.offspring_size, self.n))
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
        gaussian_vector = rng.standard_normal((self.offspring_size, self.n))
        mutated_solutions = solutions + parameters*gaussian_vector
        return mutated_solutions

    def select_survivors(self, offspring):
        """
        Deterministic and elitist survivor selection by
        replacing the current individuals by the µ best
        individuals from the offspring, a.k.a, a (µ,λ) scheme.
        """
        # Compute the fitness for each child and select the best ones.
        offspring_fitness = np.array([self.fitness(i) for i, _ in offspring])
        best_offspring_indices = np.argsort(offspring_fitness)[-self.population_size:]
        self.population = offspring[best_offspring_indices]
        self.pop_fitness = offspring_fitness[best_offspring_indices]
