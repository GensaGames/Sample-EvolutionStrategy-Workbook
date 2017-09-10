import numpy as np
import sys
import getopt
import string
import time


def main(argv):
    try:
        opts, args = getopt.getopt(argv, 'w:p:', ['words=', 'population='])
    except getopt.GetoptError:
        sys.exit(2)

    words, population = None, None
    for opt, arg in opts:
        if opt in ('-w', '--words'):
            words = arg
        if opt in ('-p', '--population'):
            population = int(arg)

    if words is None or population is None:
        print "Wrong entered values. "
        return

    start_learning(words, population)


###############################################
def random_chars(count):
    chars = ""
    for i in range(count):
        r_char = string.letters[np.random
            .randint(len(string.letters))]
        chars = chars + r_char
    return chars


def print_population_info(population, generation):
    print "------------"
    print "Generation: " + str(generation)
    for gene in population:
        print gene.gene_words(),
    print ""
    print "------------"


def start_learning(looking_words, size):
    population = []
    for i in range(size):
        gene = Gene(random_chars(len(looking_words)))
        population.append(gene)
    continue_learning(looking_words, population, 0)


def continue_learning(looking_words, population, generation):
    for gene in population:
        gene.calc_cost(looking_words)
    population.sort(key=lambda x: x.cost)
    print_population_info(population, generation)

    # Mating most efficient genes, to create
    # and calculate new updated genes. By default using Step 2.
    STEP_MOVING = 2

    updated_genes = population[0].mate(population[1])
    for gene in updated_genes:
        gene.calc_cost(looking_words)
    population = population[:-STEP_MOVING] + updated_genes
    population.sort(key=lambda x: x.cost)

    # Mutate all other Genes, instead most efficient.
    # Using 100% mutation for single chars in Genes
    MUTATE_STEP = 1

    for index in range(len(population)):
        gene = population[index]
        if index >= STEP_MOVING:
            gene.mutate(MUTATE_STEP)
            gene.calc_cost(looking_words)

        if gene.is_done(looking_words):
            print_population_info(population, generation)
            sys.exit(2)

    TIME_SLEEP = 0.1
    time.sleep(TIME_SLEEP)
    continue_learning(looking_words, population, generation + 1)


###############################################
class Gene:
    def __init__(self, words):
        self.words = words
        self.cost = None
        self.costly_index = None

    def mutate(self, chance):
        if np.random.sample() > chance:
            return
        index = self.costly_index
        r_char = string.letters[np.random.randint(len(string.letters))]

        if index is None:
            index = np.random.randint(len(self.words))
        self.words = self.words.replace(self.words[index], r_char, 1)

    def mate(self, other_gene):
        pivot = len(self.words) / 2

        child1 = self.words[:pivot] + other_gene.words[pivot:]
        child2 = other_gene.words[:pivot] + self.words[pivot:]
        return [Gene(child1), Gene(child2)]

    def calc_cost(self, source_words):
        most_costly_value = 0
        total_cost = 0
        for i in range(len(self.words)):
            index_cost = abs(ord(self.words[i]) - ord(source_words[i]))

            if index_cost > most_costly_value:
                most_costly_value = index_cost
                self.costly_index = i
            total_cost += index_cost

        self.cost = total_cost

    def gene_words(self):
        return self.words

    def is_done(self, source_words):
        return self.words == source_words


if __name__ == "__main__":
    main(sys.argv[1:])
