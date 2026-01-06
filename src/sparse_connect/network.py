import numpy as np
import random
import logging
import math

class Node:
    def __init__(self, number_of_nodes=0, connectivity_average=0, connectivity_std_dev=0, prob_arr=None):
        self.prob_arr = prob_arr
        self.number_of_nodes = number_of_nodes
        if self.prob_arr is None:
            if number_of_nodes < 1 or connectivity_average <= 0:
                raise ValueError(f"Node.__init__(): prob_arr is None and number_of_nodes ({number_of_nodes}) < 1 or connectivity_average ({connectivity_average}) <= 0")
            self.prob_arr = np.zeros((number_of_nodes, number_of_nodes))
            connectivity = connectivity_average + connectivity_std_dev * np.random.randn()
            connectivity = round(connectivity)
            connectivity = max(connectivity, 2)  # At least two out-connections
            connectivity = min(connectivity, self.number_of_nodes)  # Not more than N connections
            neighbors = random.sample(range(self.number_of_nodes), connectivity)  # [1, 4, 5, ...]
            prob = 1.0 / len(neighbors)  # Prior: Uniform distribution

            for routed_node in neighbors:
                self.prob_arr[:, routed_node] = prob

        else:  # prob_arr is not None
            shape = self.prob_arr.shape
            if len(shape) != 2:
                raise ValueError(f"Node.__init__(): len(shape ({len(shape)}) != 2")
            if shape[0] != shape[1]:
                raise ValueError(f"Node.__init__(): prob_arr is not square ({shape})")
            self.number_of_nodes = shape[0]


    def routing_node(self, target_node):
        if target_node < 0 or target_node >= self.number_of_nodes:
            raise ValueError(f"Node.routing_node(): target_node = {target_node}: target_node < 0 or target_node >= {self.number_of_nodes}")
        index_prob_list = list(zip(np.arange(self.number_of_nodes), self.prob_arr[target_node, :]))
        return int(roulette(index_prob_list))

    def normalize_row(self, row_ndx):
        row_sum = self.prob_arr[row_ndx, :].sum()
        self.prob_arr[row_ndx, :] = self.prob_arr[row_ndx, :]/row_sum

    def add_to_prob(self, target_node, routed_node, delta):
        self.prob_arr[target_node, routed_node] = np.clip(self.prob_arr[target_node, routed_node] + delta, 0, 1)
        self.normalize_row(target_node)

    def collapse(self):  # Set the probability to 1 for the highest probability, for each target node
        max_ndx = self.prob_arr.argmax(axis=1)  # (N_nodes)
        one_hot = np.zeros_like(self.prob_arr)  # (N_nodes, N_nodes)
        one_hot[np.arange(self.prob_arr.shape[0]), max_ndx] = 1
        self.prob_arr = one_hot



def roulette(index_prob_list):
    running_sum = 0
    rdm_nbr = random.random()
    for ndx, p in index_prob_list:
        running_sum += p
        if running_sum >= rdm_nbr:
            return ndx
    raise ValueError(f"roulette(): With index_prob_list = {index_prob_list}), we reached the end of the loop")

class Network:
    def __init__(self, number_of_nodes=10, connectivity_average=3, connectivity_std_dev=0, gain_range=[0.8, 1.0],
                 maximum_hops=100, maximum_hops_penalty=0.1):
        self.number_of_nodes = number_of_nodes
        self.connectivity_average = connectivity_average
        self.connectivity_std_dev = connectivity_std_dev
        self.gain_range = gain_range
        self.maximum_hops = maximum_hops
        self.maximum_hops_penalty = maximum_hops_penalty
        self.nodes = []
        for node_ndx in range(self.number_of_nodes):
            self.nodes.append(Node(self.number_of_nodes, self.connectivity_average, self.connectivity_std_dev))
        # Random gain: uniform an a log scale
        min_log_gain = math.log(self.gain_range[0])
        max_log_gain = math.log(self.gain_range[1])
        log_gain_arr = min_log_gain + (max_log_gain - min_log_gain) * np.random.rand(self.number_of_nodes, self.number_of_nodes)
        self.gain_arr = np.exp(log_gain_arr)

    def hops(self, starting_node, target_node):
        visited_nodes = [starting_node]
        gains = []
        current_node = starting_node
        while len(visited_nodes) < self.maximum_hops:
            routed_node = self.nodes[current_node].routing_node(target_node)
            visited_nodes.append(routed_node)
            gains.append( float(self.gain_arr[current_node, routed_node]) )
            current_node = routed_node
            if current_node == target_node:
                return visited_nodes, gains
        return visited_nodes, gains

    def update_probabilities(self, visited_nodes, gains, target_node, learning_rate=1.0):
        deserve_reward = True
        if len(visited_nodes) >= self.maximum_hops and visited_nodes[-1] != target_node:
            deserve_reward = False
        logging.debug(f"Network.update_probabilities(): deserve_reward = {deserve_reward}")
        probs = []
        for origin_ndx in range(len(visited_nodes) - 1):
            origin_node = visited_nodes[origin_ndx]
            dest_node = visited_nodes[origin_ndx + 1]
            probs.append( float(self.nodes[origin_node].prob_arr[target_node, dest_node]))
        logging.debug(f"Network.update_probabilities(): probs = {probs}")
        for origin_ndx in range(len(visited_nodes) - 1):
            origin_node = visited_nodes[origin_ndx]
            routed_node = visited_nodes[origin_ndx + 1]
            logging.debug(f"Network.update_probabilities(): origin_ndx = {origin_ndx}; origin_node = {origin_node}; routed_node = {routed_node}")
            reward = learning_rate
            for hop_ndx in range(origin_ndx, len(visited_nodes) - 1):
                reward *= probs[hop_ndx] * gains[hop_ndx]
                logging.debug(f"Network.update_probabilities(): probs[hop_ndx] = {probs[hop_ndx]}; gains[hop_ndx] = {gains[hop_ndx]}")
            logging.debug(f"Network.update_probabilities(): reward = {reward}")
            if deserve_reward:
                self.nodes[origin_node].add_to_prob(target_node, routed_node, reward)
            else:
                self.nodes[origin_node].add_to_prob(target_node, routed_node, -self.maximum_hops_penalty)

    def collapse(self):  # Collapse each node, into a one-hot vector (winner takes all on probabilities)
        for node in self.nodes:
            node.collapse()

    def get_adjacency_matrix(self, target_node):
        """
        Returns an NxN matrix where matrix[i, j] is the probability of node i
        routing to node j for a given target_node.
        """
        matrix = np.zeros((self.number_of_nodes, self.number_of_nodes))
        for i, node in enumerate(self.nodes):
            matrix[i, :] = node.prob_arr[target_node, :]
        return matrix
