import random
import networkx as nx
import numpy as np

# created graph manually without networkx, because it will allow to create graph faster
def create_graph(num_nodes, num_edges):
    if num_edges < (num_nodes - 1):
        raise Exception("The number of edges is less than the number of nodes - 1, we cannot get connected graph")

    weight_matrix = [[-1 for _ in range(num_nodes)] for _ in range(num_nodes)]
    connected_nodes = set()

    for edge_num in range(num_edges):
        if edge_num == 0:
            # randomly choose first 2 nodes/vertices (their numbers)
            node_one = random.randint(0, num_nodes - 1)
            node_two = random.randint(0, num_nodes - 1)
            # we need to ensure the nodes (their numbers) are different
            while node_one == node_two:
                node_two = random.randint(0, num_nodes - 1)

            # randomly generate weight of an edge
            weight = random.randint(1, 10)
            weight_matrix[node_one][node_two] = weight
            weight_matrix[node_two][node_one] = weight

            # add the numbers of 2 first nodes to the set of the nodes that are already connected
            connected_nodes.add(node_one)
            connected_nodes.add(node_two)
        else:
            # list of the nodes that are not yet connected in the graph
            unconnected = [n for n in range(num_nodes) if n not in connected_nodes]

            if unconnected:
                # if we have such nodes we choose randomly one of them
                new_node = random.choice(unconnected)
                # and connect it with already connected node
                # thus we ensure our graph will be connected with nodes from number 0 to num_nodes - 1
                existing_node = random.choice(list(connected_nodes))

                # randomly generate weight of an edge
                weight = random.randint(1, 10)
                weight_matrix[new_node][existing_node] = weight
                weight_matrix[existing_node][new_node] = weight

                connected_nodes.add(new_node)
            else:
                # if all the nodes are already connected than we create an edge between two randomly chosen of them
                node_one = random.choice(list(connected_nodes))
                node_two = random.choice(list(connected_nodes))
                while node_one == node_two:
                    node_two = random.choice(list(connected_nodes))

                weight = random.randint(1, 10)
                weight_matrix[node_one][node_two] = weight
                weight_matrix[node_two][node_one] = weight

    # create networkx graph and add weighted edges
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            w = weight_matrix[i][j]
            if w != -1:
                g.add_edge(i, j, weight=w)

    return g

# custom graph environment
class GraphEnv:
    def __init__(self, our_graph, max_steps=100):
        self.num_steps_taken = 0
        self.current_node = None
        self.our_graph = our_graph
        self.num_nodes = our_graph.number_of_nodes()
        self.start_node = 0 # 0
        self.goal_node = self.num_nodes - 1
        # the maximum number of steps agent can take
        self.max_steps = max_steps

        # adjacency matrix
        self.adjacency_matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=int)
        # weight matrix
        self.weight_matrix = np.full((self.num_nodes, self.num_nodes), -1.0, dtype=float)

        for (u, v) in self.our_graph.edges():
            w = self.our_graph[u][v]['weight']
            self.adjacency_matrix[u, v], self.adjacency_matrix[v, u] = 1, 1
            self.weight_matrix[u, v], self.weight_matrix[v, u] = w, w

        self.reset()

    def reset(self):
        self.current_node = self.start_node
        self.num_steps_taken = 0
        return self._get_observation()

    def _get_observation(self):
        # return the list of weights of edges from current_node to others, if there is no edge, then -1
        return self.weight_matrix[self.current_node].copy()

    def get_valid_actions(self):
        # return the list of nodes/vertices (their numbers) which are connected to the current_node
        return np.where(self.adjacency_matrix[self.current_node] == 1)[0]

    def step(self, action):
        self.num_steps_taken += 1
        done = False
        # info = {}
        cost = self.weight_matrix[self.current_node, action]
        # reward = -float(cost)
        custom_reward = -float(cost)
        # go to the chosen next node/vertice
        self.current_node = action
        # info["invalid"] = False

        # if the agent reached the goal node we add the specified number
        if self.current_node == self.goal_node:
            # custom_reward += 100.0
            custom_reward += 2 * nx.dijkstra_path_length(self.our_graph, self.start_node, self.goal_node, weight='weight')
            done = True

        # if we did not find a goal node in max_steps
        if self.num_steps_taken >= self.max_steps:
            done = True

        return self._get_observation(), custom_reward, done # , info

def perturb_graph(graph, weight_change=10, num_changes=20):
    # copy initial graph and return the graph with changed weighted edges
    perturb_g = graph.copy()
    edges = list(perturb_g.edges())
    changed_edges = set()
    # randomly change weights of the defined number of edges
    for _ in range(num_changes):
        u, v = random.choice(edges)
        while (u, v) in changed_edges or (v, u) in changed_edges:
              u, v = random.choice(edges)
        perturb_g[u][v]['weight'] = max(1, perturb_g[u][v]['weight'] + random.choice([-weight_change, weight_change]))
        changed_edges.add((u, v))
        changed_edges.add((v, u))
    return perturb_g


def perturb_graph_struct(graph, num_struct_changes=3):
    new_g = graph.copy()
    n = new_g.number_of_nodes()

    # our edges of the graph
    edges = list(new_g.edges())

    changes_done = 0
    while changes_done < num_struct_changes:
        (u, v) = random.choice(edges)
        old_weight = new_g[u][v]['weight']

        new_g.remove_edge(u, v)

        # find the possible pairs of nodes that are not yet connected
        possible_pairs = []
        for a in range(n):
            for b in range(a + 1, n):
                if not new_g.has_edge(a, b):
                    possible_pairs.append((a, b))

        if not possible_pairs:
            new_g.add_edge(u, v, weight=old_weight)
            break

        x, y = random.choice(possible_pairs)
        new_weight = random.randint(1, 10)
        new_g.add_edge(x, y, weight=new_weight)

        # if the graph remains connected after adding new edge and removing previous
        # else graph is reverted to the previous state of the graph
        if nx.is_connected(new_g):
            edges = list(new_g.edges())
            changes_done += 1
        else:
            new_g.remove_edge(x, y)
            new_g.add_edge(u, v, weight=old_weight)

    return new_g