import math
import numpy as np


class Terms:

    def __init__(self):
        self.attribute = ''
        self.value = ''
        self.pheromone = None
        self.heuristic = None
        self.entropy = None
        self.logK_entropy = None
        self.probability = None
        self.term_idx = None

    def set_entropy(self, dataset):

        # A POSTERIORI PROBABILITY: P(W|A=V)
        class_idx = dataset.col_index[dataset.class_attr]
        attr_idx = dataset.col_index[self.attribute]
        data = dataset.data
        rows = list(np.where(data[:, attr_idx] == self.value)[0])  # list of row index where < self.attr = self.value >
        term_freq = len(rows)

        class_freq = {}.fromkeys(dataset.class_values, 0)
        for r in rows:
            class_freq[data[r, class_idx]] += 1

        # ENTROPY
        if term_freq > 0:
            entropy = 0
            for w in class_freq:
                prob_posteriori = class_freq[w]/term_freq
                if prob_posteriori != 0:
                    entropy -= prob_posteriori * math.log2(prob_posteriori)
            self.entropy = entropy
        else:
            print('Heuristic calc error: Term value doesnt appear in current dataset')

        self.logK_entropy = math.log2(len(dataset.class_values)) - self.entropy

        return

    def set_heuristic(self, k, denominator):

        if float(denominator) == 0.0:   # CHECK THE OCCURRENCE OF THIS CONDITION <<<<<<<<
            denominator = 0.0000001

        log_k = math.log2(k)
        fnc_heuristic = (log_k - self.entropy) / denominator
        self.heuristic = fnc_heuristic

        return

    def set_pheromone(self, data):
        return

    def set_probability(self, denominator):

        if denominator == 0.0:
            self.probability = 0
        else:
            self.probability = (self.heuristic * self.pheromone) / denominator

        return

