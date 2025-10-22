from typing import Literal

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Subset

import GPy
import random
import igraph as ig
import pandas as pd
import uuid
import math
import os
from torch.distributions import MultivariateNormal, Normal, Laplace, Gumbel

import networkx as nx
import json


# Different type of functions
class LinearFunction(nn.Module):
    def __init__(self, function_params, features_per_node):
        super().__init__()
        self.in_dim = function_params['in_dim']
        self.linear = nn.Linear(self.in_dim * features_per_node, 1 * features_per_node)  # Projects from input_dim -> 1

    def forward(self, x):
        return self.linear(x).squeeze(-1)


class GPSampler(nn.Module):
    def __init__(self, f_magn=1.0, lengthscale=1.0, jitter=1e-6):
        super().__init__()
        self.f_magn = f_magn
        self.lengthscale = lengthscale
        self.jitter = jitter

    def rbf_kernel(self, X1, X2):
        # Compute pairwise squared Euclidean distances
        X1_sq = X1.pow(2).sum(dim=1, keepdim=True)
        X2_sq = X2.pow(2).sum(dim=1, keepdim=True)
        dist_sq = X1_sq - 2 * X1 @ X2.T + X2_sq.T

        # Compute RBF kernel
        K = self.f_magn * torch.exp(-0.5 * dist_sq / self.lengthscale ** 2)
        return K

    def forward(self, X):
        """
        Sample from a GP prior with RBF kernel given input X.
        X: Tensor of shape [n, d]
        Returns: Tensor of shape [n]
        """
        K = self.rbf_kernel(X, X)
        K += self.jitter * torch.eye(K.size(0), device=K.device)  # for numerical stability

        # Sample from multivariate normal
        L = torch.linalg.cholesky(K)
        z = torch.randn(X.size(0), device=X.device)
        sample = L @ z
        return sample


class MLPFunction(nn.Module):
    def __init__(self, in_dim, out_dim=1, init_std=1, device='cpu', features_per_node=1):
        '''
        params:
            d: input/output dimension
            sigma: std of noise
        '''
        super(MLPFunction, self).__init__()
        activation_choices = {
            "identity": nn.Identity(),
            "sigmoid": nn.Hardsigmoid(),
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "softplus": nn.Softplus()
        }
        self.in_dim = in_dim["in_dim"]
        self.out_dim = out_dim
        self.selected_name, self.activation = random.choice(list(activation_choices.items()))

        self.mlp = nn.Sequential(
            *[nn.Linear(self.in_dim * features_per_node, self.in_dim * features_per_node, device=device),
              self.activation,
              nn.Linear(self.in_dim * features_per_node, self.out_dim * features_per_node, device=device),
              self.activation
              ])
        for i, (n, p) in enumerate(self.mlp.named_parameters()):
            if len(p.shape) == 2:
                nn.init.normal_(p, std=init_std)

    def forward(self, x):
        with torch.no_grad():
            output = self.mlp(x)
        return output.float()


class Dist(object):
    def __init__(self,
                 d,
                 noise_std=1,
                 noise_type='Gauss',
                 adjacency=None,
                 function_type: Literal['Linear', 'GP', 'MLP'] = 'Linear',
                 function_params=None, features_per_node=1):
        self.d = d
        if isinstance(noise_std, (int, float)):
            noise_std = noise_std * torch.ones(self.d * features_per_node)
        self.function_type = function_type
        self.function_params = function_params

        ## Graph
        self.adjacency = adjacency
        if adjacency is None:
            self.adjacency = np.ones((d, d))
            self.adjacency[np.tril_indices(d)] = 0

        # Needs strictly upper triangular matrix
        assert (np.allclose(self.adjacency, np.triu(self.adjacency)))
        self.features_per_node = features_per_node
        ## Noise
        if noise_type == 'Gauss':
            self.noise = Normal(0, noise_std)  # give standard deviation
        elif noise_type == 'Laplace':
            self.noise = Laplace(0, noise_std / np.sqrt(2))
        elif noise_type == 'Gumbel':
            self.noise = Gumbel(0, np.sqrt(noise_std) * np.sqrt(6) / np.pi)
        else:
            raise NotImplementedError("Unknown noise type.")

    def sample(self, n):
        noise = self.noise.sample((n,))  # n x d noise matrix
        X = torch.zeros(n, self.d * self.features_per_node)

        # !!! Only works if adjacency matrix is upper triangular !!!
        for i in range(0, self.d * self.features_per_node, self.features_per_node):
            parents = np.nonzero(self.adjacency[:, i // self.features_per_node])[0]
            # X[:, i:i + self.features_per_node] = noise[:, i:i + self.features_per_node]
            parents = [[parentN for parentN in range(parent * self.features_per_node,
                                                     parent * self.features_per_node + self.features_per_node)] for
                       parent in parents]
            if len(parents) > 0:
                # Generate an assignment instance for each node
                if self.function_type == 'Linear':
                    self.function_params['in_dim'] = len(parents)
                    assignment = LinearFunction(self.function_params, self.features_per_node)
                elif self.function_type == 'GP':
                    assignment = GPSampler(self.function_params, self.features_per_node)
                elif self.function_type == 'MLP':
                    self.function_params['in_dim'] = len(parents)
                    assignment = MLPFunction(self.function_params, features_per_node=self.features_per_node)

                X_par = X[:, parents].view(n, -1)
                X[:, i:i + self.features_per_node] += (torch.ones_like(noise[:, i:i + self.features_per_node]) - noise[
                                                                                                                 :,
                                                                                                                 i:i + self.features_per_node]) * torch.tensor(
                    assignment(X_par))  # Additive noise model
            else:
                X[:, i:i + self.features_per_node] = noise[:, i:i + self.features_per_node]
        return X


class MulticlassRank(nn.Module):
    '''多类排序任务'''

    def __init__(self, num_classes):
        super().__init__()

        def class_sampler_f(min_, max_):
            def s():
                if random.random() > 0.5:
                    uniform_int_sampler_f = lambda a, b: lambda: round(np.random.uniform(a, b))
                    return uniform_int_sampler_f(min_, max_)()
                return 2

            return s

        # self.num_classes = class_sampler_f(2, num_classes)()
        self.num_classes = 2

    def forward(self, x, class_boundaries=None):
        # x has shape (T,B,H)

        # CAUTION: This samples the same idx in sequence for each class boundary in a batch
        if class_boundaries is None:
            class_boundaries = torch.randint(0, x.shape[0], (self.num_classes - 1,))
            class_boundaries = x[class_boundaries].unsqueeze(1)

        d = (x > class_boundaries).sum(axis=0)

        return d, class_boundaries


def simulate_dag(d, s0, graph_type, triu=False):
    """Simulate random DAG with some expected number of edges.
    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF

    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """

    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        if triu:
            return np.triu(B_und, k=1)
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=False)
        B_und = _graph_to_adjmat(G)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    else:
        raise ValueError('unknown graph type')
    if not triu:
        B = _random_permutation(B)
    assert ig.Graph.Adjacency(B.tolist()).is_dag()
    return B


def generate_dag_with_max_indegree(n, max_indegree, triu=False):
    # Step 1: Create a topological ordering of nodes
    topo_order = list(range(n))
    random.shuffle(topo_order)

    edges = []

    # Step 2: For each node, randomly choose ≤ k predecessors
    for i in range(1, n):
        target = topo_order[i]
        # Eligible sources are earlier in topological order
        candidates = topo_order[:i]
        num_parents = min(max_indegree, len(candidates))
        num_selected = random.randint(0, num_parents)
        sources = random.sample(candidates, num_selected)
        for src in sources:
            edges.append((src, target))

    # Step 3: Create the DAG
    g = ig.Graph(n=n, edges=edges, directed=True)
    B = np.array(g.get_adjacency().data)
    return B


def generate_data(d, s0, N,
                  noise_std=0.1,
                  noise_type='Gauss',
                  graph_type='ER',
                  function_type: Literal['Linear', 'GP', 'MLP'] = 'MLP',
                  function_params=dict(), features_per_node=1):
    adjacency = simulate_dag(d, s0, graph_type, triu=True)
    teacher = Dist(d, noise_std=noise_std, noise_type=noise_type, adjacency=adjacency, function_type=function_type,
                   function_params=function_params, features_per_node=features_per_node)
    X = teacher.sample(N)
    return X, adjacency


def sample_y_index(adjacency, max_parent_y):
    while True:
        index_y = np.random.choice(len(adjacency))  # randomly pick a node
        parents = np.nonzero(adjacency[:, index_y])[0]  # find its parents (incoming edges)

        if 0 < len(parents) <= max_parent_y:
            return index_y


def get_parents(adj_matrix, node_index):
    return list(np.where(adj_matrix[:, node_index] == 1)[0])


def get_grandparents(adj_matrix, node_index):
    parents = get_parents(adj_matrix, node_index)
    grandparents = set()
    for p in parents:
        gp = get_parents(adj_matrix, p)
        grandparents.update(gp)
    return list(grandparents)


def get_children(adj_matrix, node_index):
    return list(np.where(adj_matrix[node_index, :] == 1)[0])


def get_coparents(adj_matrix, node_index):
    target_children = set(get_children(adj_matrix, node_index))
    coparents = set()

    for other_node in range(adj_matrix.shape[0]):
        if other_node == node_index:
            continue
        other_children = set(get_children(adj_matrix, other_node))
        if target_children & other_children:  # non-empty intersection
            coparents.add(other_node)

    return list(coparents)


def get_markov_blanket(adj_matrix, node_index):
    parents = set(get_parents(adj_matrix, node_index))
    children = set(get_children(adj_matrix, node_index))
    coparents = set()

    for child in children:
        coparents.update(get_parents(adj_matrix, child))

    # Remove the node itself if it's in the co-parents
    coparents.discard(node_index)

    blanket = parents | children | coparents
    return list(blanket)


def get_siblings(adj_matrix, node_index):
    parents = get_parents(adj_matrix, node_index)
    siblings = set()

    for parent in parents:
        children = get_children(adj_matrix, parent)
        siblings.update(children)

    siblings.discard(node_index)  # remove self
    return list(siblings)


def get_new_indices_after_removal(original_indices, removed_index):
    reduced = [i for i in original_indices if i != removed_index]
    new_index_map = {orig: new_idx for new_idx, orig in enumerate(reduced)}
    return new_index_map


def map_original_to_permuted(original_list, index_mapping):
    return [index_mapping[i].item() for i in original_list]


def generate_datasets(d, s0, N, n_datasets,
                      num_classes=2,
                      train_test_split=0.7,
                      max_parent_y=3,
                      noise_std=0.1,
                      noise_type='Gauss',
                      graph_type='ER',
                      function_type: Literal['Linear', 'GP', 'MLP'] = 'Linear',
                      function_params=dict(), features_per_node=1):
    datasets = []
    for i in range(n_datasets):
        data_name = 'dataset_{}'.format(i)
        data, adjacency = generate_data(d, s0, N, noise_std, noise_type, graph_type, function_type, function_params,
                                        features_per_node)

        y_idx = sample_y_index(adjacency, max_parent_y) * features_per_node
        y = data[:, y_idx]
        train_idx = round(len(data) * train_test_split)
        class_assigner = MulticlassRank(num_classes)
        y_train_discrete, class_boundries_train = class_assigner(y[:train_idx])
        y_test_discrete, class_boundries = class_assigner(y[train_idx:], class_boundries_train)
        y_discrete = torch.cat(((y_train_discrete, y_test_discrete)))
        # y_discrete=y
        x = torch.cat((data[:, :y_idx], data[:, y_idx + features_per_node:]), dim=1)
        # perm = torch.randperm(x.size(1))
        # x_permuted = x[:, perm]
        #
        # x_new_idx = get_new_indices_after_removal(np.arange(adjacency.shape[0]), y_idx)
        # for k, v in x_new_idx.items():
        #     x_new_idx[k] = torch.where(perm == v)[0]
        #
        # causal_info = {'parents':map_original_to_permuted(get_parents(adjacency, y_idx), x_new_idx),
        #                'children':map_original_to_permuted(get_children(adjacency, y_idx), x_new_idx),
        #                'coparents':map_original_to_permuted(get_coparents(adjacency, y_idx), x_new_idx),
        #                'siblings':map_original_to_permuted(get_siblings(adjacency, y_idx), x_new_idx),
        #                'coparents':map_original_to_permuted(get_coparents(adjacency, y_idx), x_new_idx),
        #                'grandparents':map_original_to_permuted(get_grandparents(adjacency, y_idx), x_new_idx),
        #                'markov_blanket':map_original_to_permuted(get_markov_blanket(adjacency, y_idx), x_new_idx)}

        # datasets.append((data_name, x_permuted, y_discrete, causal_info))
        datasets.append((data_name, x, y_discrete, adjacency, y_idx // features_per_node))

    return datasets


# d=10
# s0=10
# N=10
# X, adjacency = generate_data(d, s0, N)
# print(X)
# n_datasets = 1
# datasets = generate_datasets(d, s0, N, n_datasets)

# X, adjacency = generate_data(d, s0, N, noise_std=1, lengthscale=0.1)
# dag = nx.from_numpy_array(adjacency, create_using=nx.DiGraph)

# plt.figure(figsize=(8, 4))
# nx.draw(dag, with_labels=True, node_color='lightblue', edge_color='gray')
# plt.savefig('dag.png')
# print(adjacency)
# print(X)

if __name__ == '__main__':
    d = 11
    s0 = 10
    N = 1000
    # X, adjacency = generate_data(d, s0, N)
    # print(X)
    n_datasets = 1
    # datasets = generate_datasets(d, s0, N, n_datasets)
    datasets = generate_datasets(d, s0, N, n_datasets, features_per_node=2)
    datasets
    # X, adjacency = generate_data(d, s0, N, noise_std=1, lengthscale=0.1)
    # dag = nx.from_numpy_array(adjacency, create_using=nx.DiGraph)
    #
    # plt.figure(figsize=(8, 4))
    # nx.draw(dag, with_labels=True, node_color='lightblue', edge_color='gray')
    # plt.savefig('dag.png')
