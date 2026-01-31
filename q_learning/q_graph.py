import numpy as np
import random
import pandas as pd

class QNode:
    def __init__(self, number_of_nodes=0, connectivity_average=0, connectivity_std_dev=0, Q_arr=None, neighbor_nodes=None,
                 state_dict=None):
        if state_dict is not None:
            self.Q = state_dict['Q']
            self.number_of_nodes = state_dict['number_of_nodes']
            self.neighbor_nodes = state_dict['neighbor_nodes']
        else:  # state_dict is None
            if Q_arr is None:
                self.number_of_nodes = number_of_nodes
                number_of_neighbors = connectivity_average + connectivity_std_dev * np.random.randn()
                number_of_neighbors = round(number_of_neighbors)
                number_of_neighbors = max(number_of_neighbors, 2)  # At least two out-connections
                number_of_neighbors = min(number_of_neighbors, self.number_of_nodes)  # Not more than N connections
                self.neighbor_nodes = random.sample(range(self.number_of_nodes), number_of_neighbors)  # [1, 4, 5, ...]
                self.Q = np.zeros((self.number_of_nodes, number_of_neighbors))  # Optimistic initialization: all rewards will be negative
                # q = self.Q[state, action]: state = target node; action = chosen neighbor node (converted to column index) to route the message to

            else:  # state_dict is None and Q_arr is not None
                self.Q = Q_arr
                self.number_of_nodes = self.Q.shape[0]
                self.neighbor_nodes = neighbor_nodes

    def neighbor_column(self, node):
        if not node in self.neighbor_nodes:
            raise ValueError(f"QNode.neighbor_column(): node {node} is not part of the neighbors ({self.neighbor_nodes})")
        return self.neighbor_nodes.index(node)

    def neighbor_node(self, column_index):
        return self.neighbor_nodes[column_index]

    def epsilon_greedy(self, target_node, epsilon):
        rdm_nbr = random.random()
        if rdm_nbr < epsilon:  # Random choice
            random_choice = random.choice(self.neighbor_nodes)
            return random_choice
        else:  # Greedy choice
            neighbor_columns = np.where(self.Q[target_node, :] == self.Q[target_node, :].max())[0]  # [1, 4, 5]
            neighbor_column = random.choice(neighbor_columns)
            neighbor_node = self.neighbor_node(neighbor_column)
            return neighbor_node

    def to_dict(self):
        state_dict = {}
        state_dict['number_of_nodes'] = self.number_of_nodes
        state_dict['neighbor_nodes'] = self.neighbor_nodes
        state_dict['Q'] = self.Q
        return state_dict


class QGraph:
    def __init__(self, number_of_nodes=10, connectivity_average=3, connectivity_std_dev=0, cost_range=[0.0, 1.0],
                 maximum_hops=100, maximum_hops_penalty=1.0):
        self.number_of_nodes = number_of_nodes
        self.connectivity_average = connectivity_average
        self.connectivity_std_dev = connectivity_std_dev
        self.cost_range = cost_range
        self.maximum_hops = maximum_hops
        self.maximum_hops_penalty = maximum_hops_penalty
        self.QNodes = []
        for node in range(self.number_of_nodes):
            self.QNodes.append(QNode(self.number_of_nodes, self.connectivity_average, self.connectivity_std_dev))

        self.cost_arr = cost_range[0] + (cost_range[1] - cost_range[0]) * np.random.random((self.number_of_nodes, self.number_of_nodes))

    def trajectory(self, start_node, target_node, epsilon):
        visited_nodes = [start_node]
        costs = []
        if start_node == target_node:
            return visited_nodes, costs
        current_node = start_node
        while len(visited_nodes) < self.maximum_hops + 1:
            next_node = self.QNodes[current_node].epsilon_greedy(target_node, epsilon)
            cost = float(self.cost_arr[current_node, next_node])
            visited_nodes.append(next_node)
            costs.append(cost)
            current_node = next_node
            if current_node == target_node:
                return visited_nodes, costs
        # We reached the maximum number of hops
        return visited_nodes, costs

    def update_Q_from_trajectory(self, visited_nodes, costs, alpha, gamma, target_node):
        if len(visited_nodes) != len(costs) + 1:
            raise ValueError(f"QGraph.update_Q(): len(visited_nodes) ({len(visited_nodes)}) != len(costs) + 1 ({len(costs)} + 1)")
        node_pairs = list(zip(visited_nodes, visited_nodes[1:]))  # [(9, 3), (3, 4), ... (5, 7)]
        for pair_ndx in range(len(node_pairs)):
            origin_node, dest_node = node_pairs[pair_ndx]
            cost = costs[pair_ndx]
            reward = -cost
            if visited_nodes[-1] != target_node:
                reward = -self.maximum_hops_penalty
            # Q_orig(target, dest) <- (1 - alpha) Q_orig(target, dest) + alpha * ( r + gamma * max_neigh' Q_dest(target, neigh') )
            Q_orig_target_dest = self.QNodes[origin_node].Q[target_node, self.QNodes[origin_node].neighbor_column(dest_node) ]
            max_neigh_Q_dest_target_neigh = np.max(self.QNodes[dest_node].Q[target_node, :])
            updated_Q = (1 - alpha) * Q_orig_target_dest + alpha * (reward + gamma * max_neigh_Q_dest_target_neigh)
            self.QNodes[origin_node].Q[target_node, self.QNodes[origin_node].neighbor_column(dest_node)] = updated_Q

    def update_Q(self, start_node, neighbor_node, alpha, gamma, target_node):
        cost = self.cost_arr[start_node, neighbor_node]
        reward = -cost
        # Q_orig(target, dest) <- (1 - alpha) Q_orig(target, dest) + alpha * ( r + gamma * max_neigh' Q_dest(target, neigh') )
        Q_orig_target_dest = self.QNodes[start_node].Q[target_node, self.QNodes[start_node].neighbor_column(neighbor_node)]
        max_neigh_Q_dest_target_neigh = np.max(self.QNodes[neighbor_node].Q[target_node, :])
        updated_Q = (1 - alpha) * Q_orig_target_dest + alpha * (reward + gamma * max_neigh_Q_dest_target_neigh)
        self.QNodes[start_node].Q[target_node, self.QNodes[start_node].neighbor_column(neighbor_node)] = updated_Q

    def get_adjacency_matrix(self, target_node):
        adjacency_mtx = -1 * np.ones((self.number_of_nodes, self.number_of_nodes))
        for origin_node in range(self.number_of_nodes):
            for destination_node in range(self.number_of_nodes):
                if destination_node in self.QNodes[origin_node].neighbor_nodes:
                    dest_col = self.QNodes[origin_node].neighbor_column(destination_node)
                    adjacency_mtx[origin_node, destination_node] = \
                        self.QNodes[origin_node].Q[target_node, dest_col]
            #adjacency_mtx[origin_node, :] = self.QNodes[origin_node].Q[target_node, :]
        return adjacency_mtx

    def to_dict(self):
        state_dict = {}
        state_dict['number_of_nodes'] = self.number_of_nodes
        state_dict['maximum_hops'] = self.maximum_hops
        state_dict['maximum_hops_penalty'] = self.maximum_hops_penalty
        state_dict['QNodes'] = []
        for node in self.QNodes:
            state_dict['QNodes'].append(node.to_dict())
        state_dict['cost_arr'] = self.cost_arr
        return state_dict

    def load_nodes_edges(self, nodes_df, edges_df):
        id_to_label = nodes_df.set_index("Id")["Label"].to_dict()
        #print(f"QGraph.load_nodes_edges(): id_to_label =\n{id_to_label}")
        self.number_of_nodes = len(id_to_label)
        if set(id_to_label.keys()) != set(range(self.number_of_nodes)):
            raise ValueError(f"QGraph.load_nodes_edges(): set(id_to_label.keys()) ({set(id_to_label.keys())}) != set(range(self.number_of_nodes)) ({set(range(self.number_of_nodes))}). We expect Id's to be 0, 1, ... N-1")

        edges_df_columns = edges_df.columns
        if not 'Source' in edges_df_columns or not 'Target' in edges_df_columns or not 'Weight' in edges_df_columns:
            raise ValueError(f"QGraph.load_nodes_edges(): The columns of the edged DataFrame ({edges_df_columns}) must include 'Source', 'Target', and 'Weight'")
        src_target_cost_list = list(edges_df[["Source", "Target", "Weight"]].itertuples(index=False, name=None))
        self.cost_arr = -1 * np.ones((self.number_of_nodes, self.number_of_nodes))
        self.QNodes = []
        for node in range(self.number_of_nodes):
            node_neighbor_cost_list = [(s, t, c) for (s, t, c) in src_target_cost_list if s==node]
            neighbor_nodes = [t for (s, t, c) in node_neighbor_cost_list]
            Q = np.zeros((self.number_of_nodes, len(neighbor_nodes)))
            qnode = QNode(Q_arr=Q, neighbor_nodes=neighbor_nodes)
            self.QNodes.append(qnode)
            for _, target_node, cost in node_neighbor_cost_list:
                self.cost_arr[node, target_node] = cost
