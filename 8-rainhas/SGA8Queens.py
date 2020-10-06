import numpy as np

class SGA8Queens(object):
    def __init__(self, crossover_prob=0.9, \
                    mutation_prob=0.4, \
                    pop_size=100, \
                    use_binary_rep=False, \
                    recombination_method=1, \
                    mutation_method=1, \
                    parents_sel_method=1, \
                    survivor_sel_method=1):
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

        # Use binary string representation for population.
        self.use_binary_rep = use_binary_rep

        # Set recombination method. Uses cut and crossfill for 1.
        # TODO: add other methods.
        if recombination_method == 1:
            self.recombine = self.cut_and_crossfill_crossover
        elif recombination_method == 2:
            self.recombine = self.pmx_crossover
        elif recombination_method == 3:
            self.recombine = self.edge_crossover
        elif recombination_method == 4:
            self.recombine = self.cyclic_crossover
        else:
            raise ValueError("Invalid option value for the recombination method.")
        
        # Set mutation method.
        if mutation_method == 1:
            self.mutate = self.swap_mutation
        elif mutation_method == 2:
            self.mutate = self.insert_mutation
        elif mutation_method == 3:
            self.mutate = self.scramble_mutation
        elif mutation_method == 4:
            self.mutate = self.inversion_mutation
        else:
            raise ValueError("Invalid option value for the mutation method.")
        
        # Set parent selection method. Value 1 for best 2 out of random 5, and 2
        # for roulette.
        if parents_sel_method == 1:
            self.select_parents = self.select_2_out_5
        elif parents_sel_method == 2:
            self.select_parents = self.roulette_selection
        else:
            raise ValueError("Invalid option value for the parents selection method.")

        # Set survivor selection method. Value 1 for replacement of the worst
        # strategy, and 2 for generation based approach.
        if survivor_sel_method == 1:
            self.select_survivors = self.replace_worst
        elif survivor_sel_method == 2:
            self.select_survivors = self.replace_parents_by_childs
        else:
            raise ValueError("Invalid option value for the survivor selection method.")

    # #########################
    # COMMON METHODS AND UTILS
    # #########################

    def fitness(self, table):
        # Based on https://bit.ly/32zycds.
        eps = 1
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
            if self.use_binary_rep:
                queen_pos = int(table[i], 2)
            else:
                queen_pos = table[i]
            f_row[queen_pos - 1] += 1
            f_mdiag[queen_pos + i - 1] += 1
            f_sdiag[N - queen_pos + i - 1] += 1
        
        for i in range(2*N):
            x, y, z = 0, 0, 0
            if i < N:
                x = f_row[i]
            y = f_mdiag[i]
            z = f_sdiag[i]
            result += ((x * (x - 1)) + (y * (y - 1)) + (z * (z - 1))) / 2
        
        self.num_fitness_eval += 1
        
        return 1 / (result + eps)

    def random_init_population(self):
        rng = np.random.default_rng()

        if self.use_binary_rep:
            self.population = [self._to_binary_string(rng.permutation(np.arange(1, 9))) \
                for i in range(self.pop_size)]
        else:
            self.population = [list(rng.permutation(np.arange(1, 9))) \
                for i in range(self.pop_size)]
        
        self.num_fitness_eval = 0
        self.pop_fitness = np.array([self.fitness(x) for x in self.population])
    
    def _to_binary_string(self, int_p):
        # The format string: convert the argument to a 3-bit binary filling
        # the bits to the left with zeros if unused.
        bin_p = ['{0:03b}'.format(gene) for gene in int_p]
        return bin_p

    def run(self):
        self.random_init_population()
        num_iter = 0

        while self.num_fitness_eval < 10000 and self.pop_fitness.max() < 1:
            p1, p2 = self.select_parents()
            c1, c2 = self.recombine(p1, p2)
            c1 = self.mutate(c1)
            c2 = self.mutate(c2)
            self.select_survivors(c1, c2, p1, p2)
            num_iter += 1

        report = {
            'convergence': self.pop_fitness.max() == 1, \
            'iterations': num_iter, \
            'fitness evaluations': self.num_fitness_eval, \
            'average fitness': np.average(self.pop_fitness)
        }
        
        return report
    
    # ########################

    # #########################
    # PARENT SELECTION METHODS
    # #########################

    def select_2_out_5(self):
        # Numpy's random number generator.
        rng = np.random.default_rng()
        # Get 5 different random individuals from the population.
        candidates = rng.choice(self.population, size=5, replace=False)
        # Find the two candidates with the highest fitness.
        candidates_fitness = np.array([self.fitness(x) for x in candidates])
        p1, p2 = candidates[np.argsort(-candidates_fitness)[:2]]
        p1, p2 = list(p1), list(p2)
        return p1, p2
    
    def roulette_selection(self):
        rng = np.random.default_rng()
        fitness_sum = self.pop_fitness.sum()
        parents = []

        for i in range(2):
            rand_val = rng.random()
            cum_prob = 0.0
            for f, index in zip(self.pop_fitness, np.arange(self.pop_size)):
                if rand_val < cum_prob + (f / fitness_sum):
                    parents.append(self.population[index])
                    break
                cum_prob += f / fitness_sum
        
        return parents

    # #########################

    # ######################
    # RECOMBINATION METHODS
    # ######################

    def cut_and_crossfill_crossover(self, p1, p2):
        # Implementation of a cut and crossfill crossover.
        rng = np.random.default_rng()
        c1 = 8*['']
        c2 = 8*['']

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
    
    def pmx_crossover(self, p1, p2):
        rng = np.random.default_rng()
        c1, c2 = 8*[0], 8*[0]

        if rng.uniform() < self.p_c:
            # Choose two random crossover points.
            i, j = rng.choice(8, size=2, replace=False)
            if i > j:
                i, j = j, i

            # Copy segments to children.
            c1[i:j+1], c2[i:j+1] = p1[i:j+1], p2[i:j+1]

            # For each allele in the segment...
            for k in range(i, j+1):
                # If it is not in the segment of the other parent...
                if p2[k] not in p1[i:j+1]:
                    # ... and the corresponding allele itself is not
                    # in p2's segment, then place this allele at the 
                    # position where it is in p2.
                    if p1[k] not in p2[i:j+1]:
                        c1[p2.index(p1[k])] = p2[k]
                    else:
                        # Find the next position that does not contain
                        # an allele in both segments.
                        it = p2.index(p1[k])
                        while p1[it] in p2[i:j+1]:
                            it = p2.index(p1[it])
                        it = p2.index(p1[it])
                        c1[it] = p2[k]
                # Analogous for the second child.
                if p1[k] not in p2[i:j+1]:
                    if p2[k] not in p1[i:j+1]:
                        c2[p1.index(p2[k])] = p1[k]
                    else:
                        it = p1.index(p2[k])
                        while p2[it] in p1[i:j+1]:
                            it = p1.index(p2[it])
                        it = p1.index(p2[it])
                        c2[it] = p1[k]
            
            # Fill the remaining positions.
            for k in range(8):
                if c1[k] == 0:
                    c1[k] = p2[k]
                if c2[k] == 0:
                    c2[k] = p1[k]
        else:
            c1, c2 = p1[:], p2[:]
        
        return c1, c2

    def edge_crossover(self, p1, p2):
        pass

    def cyclic_crossover(self, p1, p2):
        rng = np.random.default_rng()
        c1, c2 = 8*[0], 8*[0]

        if rng.uniform() < self.p_c:
            turn = 0
            while 0 in c1 or 0 in c2:
                start_i = c1.index(0)
                if turn == 0:
                    c1[start_i] = p1[start_i]
                    c2[start_i] = p2[start_i]
                else:
                    c1[start_i] = p2[start_i]
                    c2[start_i] = p1[start_i]
                
                curr_i = p1.index(p2[start_i])
                while curr_i != start_i:
                    if turn == 0:
                        c1[curr_i] = p1[curr_i]
                        c2[curr_i] = p2[curr_i]
                    else:
                        c1[curr_i] = p2[curr_i]
                        c2[curr_i] = p1[curr_i]
                    curr_i = p1.index(p2[curr_i])
                turn = (turn + 1) % 2
        else:
            c1, c2 = p1[:], p2[:]
        
        return c1, c2


    # #########################

    # #################
    # MUTATION METHODS
    # #################

    def swap_mutation(self, child):
        rng = np.random.default_rng()

        # Mutation by switching the position of a two genes.
        if rng.uniform() < self.p_m:
            i, j = rng.choice(8, size=2, replace=False)
            child[i], child[j] = child[j], child[i]

        return child
    
    def insert_mutation(self, child):
        rng = np.random.default_rng()

        if rng.uniform() < self.p_m:
            i, j = rng.choice(8, size=2, replace=False)
            if j < i:
                i, j = j, i
            # The idea here is to move the jth gene close to the ith gene.
            child = child[:i+1] + [child[j]] + child[i+1:j] + child[j+1:]
        
        return child
    
    def scramble_mutation(self, child):
        rng = np.random.default_rng()

        if rng.uniform() < self.p_m:
            i, j = rng.choice(8, size=2, replace=False)
            if j < i:
                i, j = j, i
            shuffled_interval = child[i:j+1]
            np.random.shuffle(shuffled_interval)
            # Scramble the genes from i to j.
            child = child[:i] + shuffled_interval + child[j+1:]
        
        return child
    
    def inversion_mutation(self, child):
        rng = np.random.default_rng()

        if rng.uniform() < self.p_m:
            i, j = rng.choice(8, size=2, replace=False)
            if j < i:
                i, j = j, i
            reversed_chunk = child[i:j+1]
            reversed_chunk.reverse()
            child = child[:i] + reversed_chunk + child[j+1:]
        
        return child

    # #########################

    # ###########################
    # SURVIVOR SELECTION METHODS
    # ###########################

    def replace_worst(self, c1, c2, p1, p2):
        # Find the two individuals with the lowest fitness value.
        weakest_solutions = np.argsort(self.pop_fitness)[:2]
        # Replace them by the childs.
        self.population[weakest_solutions[0]] = c1[:]
        self.population[weakest_solutions[1]] = c2[:]
        # Update the fitness table.
        self.pop_fitness[weakest_solutions[0]] = self.fitness(c1)
        self.pop_fitness[weakest_solutions[1]] = self.fitness(c2)

    def replace_parents_by_childs(self, c1, c2, p1, p2):
        p1_index = self.population.index(p1)
        p2_index = self.population.index(p2)
        self.population[p1_index] = c1[:]
        self.population[p2_index] = c2[:]

    # #########################
