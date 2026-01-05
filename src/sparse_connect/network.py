import numpy as np
import random

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
            #for target_node in range(self.number_of_nodes):
            """connectivity = connectivity_average + connectivity_std_dev * np.random.randn()
            connectivity = round(connectivity)
            connectivity = max(connectivity, 2)  # At least two out-connections
            neighbors = random.sample(range(self.number_of_nodes), connectivity)  # [1, 4, 5, ...]
            prob = 1.0/len(neighbors)  # Prior: Uniform distribution
            """
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


def roulette(index_prob_list):
    running_sum = 0
    rdm_nbr = random.random()
    for ndx, p in index_prob_list:
        running_sum += p
        if running_sum >= rdm_nbr:
            return ndx
    raise ValueError(f"roulette(): With index_prob_list = {index_prob_list}), we reached the end of the loop")

class Network:
    def __init__(self, number_of_nodes=10, connectivity_average=3, connectivity_std_dev=0, attenuation_range=[0.8, 1.0],
                 maximum_hops=100, maximum_hops_penalty=0.1):
        self.number_of_nodes = number_of_nodes
        self.connectivity_average = connectivity_average
        self.connectivity_std_dev = connectivity_std_dev
        self.attenuation_range = attenuation_range
        self.maximum_hops = maximum_hops
        self.maximum_hops_penalty = maximum_hops_penalty
        self.nodes = []
        for node_ndx in range(self.number_of_nodes):
            self.nodes.append(Node(self.number_of_nodes, self.connectivity_average, self.connectivity_std_dev))
        self.attenuation_arr = self.attenuation_range[0] + (self.attenuation_range[1] - self.attenuation_range[0]) * np.random.rand(self.number_of_nodes, self.number_of_nodes)

    def hops(self, starting_node, target_node):
        visited_nodes = [starting_node]
        attenuations = []
        current_node = starting_node
        while len(visited_nodes) < self.maximum_hops:
            routed_node = self.nodes[current_node].routing_node(target_node)
            visited_nodes.append(routed_node)
            attenuations.append( float(self.attenuation_arr[current_node, routed_node]) )
            current_node = routed_node
            if current_node == target_node:
                return visited_nodes, attenuations
        return visited_nodes, attenuations