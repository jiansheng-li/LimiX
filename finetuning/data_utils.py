import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import List

from utils.data_utils import cluster_test_data
from utils.retrieval_utils import find_top_K_indice


class TabularFinetuneDataset(Dataset):
    """
        A custom PyTorch Dataset for fine-tuning, supporting data shuffling and retrieval-based selection.

        This dataset prepares training and testing splits for each item. It can either shuffle the
        training data randomly or select training examples based on pre-computed attention scores
        (retrieval). For each 'step', it provides a unique training set and a corresponding test set.
        """

    def __init__(self,
                 X_train: torch.Tensor,
                 y_train: torch.Tensor,
                 X_test: torch.Tensor,
                 y_test: torch.Tensor,
                 attention_score: np.ndarray = None,
                 retrieval_len: int = 2000,
                 use_retrieval: bool = True,
                 use_cluster: bool = False,
                 cluster_num: int = None,
                 use_threshold: bool = False,
                 mixed_method: str = "max",
                 threshold: float = None
                 ):

        """
            Initializes the FinetuneDataset.
            Args:
                X_train (torch.Tensor): The full set of input training data.
                y_train (torch.Tensor): The full set of corresponding training labels.
                attention_score (np.ndarray, optional): Pre-computed attention scores for retrieval.
                                                         Shape: (num_samples_in_X_train,num_samples_in_original_X_test).
                                                         Required if use_retrieval is True.
                retrieval_len (int, optional): The number of top samples to select based on attention scores.
                                               Used only if use_retrieval is True.
                                               Note: The parameter in init_dataset is named 'train_len'.
                use_retrieval (bool, optional): Flag to determine data selection strategy.
                                                If True, uses attention scores for selection.
                                                If False, uses random shuffling.
            """
        self.init_dataset(X_train, y_train, X_test, y_test,attention_score, retrieval_len, use_retrieval, use_cluster,
                          cluster_num, use_threshold, mixed_method, threshold)

    def __len__(self):
        """
                Returns the number of steps/items in the dataset.

                Returns:
                    int: The number of steps, which corresponds to the size of the first dimension
                         of the generated X_test tensor.
                """
        return self.max_steps

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
                Retrieves a single item (a training/test split configuration) by index.

                Args:
                    idx (int): The index of the item to retrieve.

                Returns:
                    dict[str, torch.Tensor]: A dictionary containing the tensors for the training and testing splits
                                     for this specific step/index.
                                     Keys: 'X_train', 'y_train', 'X_test', 'y_test'.
                """
        return dict(
            X_train=self.X_train[idx],  # Training features for this step
            y_train=self.y_train[idx],  # Training labels for this step
            X_test=self.X_test[idx],  # Testing features for this step
            y_test=self.y_test[idx],  # Testing labels for this step
        )

    def init_dataset(self,
                     X_train: torch.Tensor,
                     y_train: torch.Tensor,
                     X_test: torch.Tensor,
                     y_test: torch.Tensor,
                     attention_score: np.ndarray = None,
                     train_len: int = 2000,
                     use_retrieval: bool =False,
                     use_cluster: bool = False,
                     cluster_num: int = None,
                     use_threshold: bool = False,
                     mixed_method: str = "max",
                     threshold: float = None
                     ):

        #TODO jianshengli confirm!
        if use_retrieval:
            if use_cluster:
                if use_threshold:
                    top_k_indices = find_top_K_indice(attention_score, threshold=threshold, mixed_method=mixed_method,retrieval_len=train_len)
                    cluster_num = min(cluster_num, len(top_k_indices))
                    cluster_train_sample_indices, cluster_test_sample_indices = cluster_test_data(top_k_indices,
                                                                                                  cluster_num)
                    self.X_train = [X_train[x_iter] for x_iter in cluster_train_sample_indices.values()]
                    self.y_train = [y_train[x_iter] for x_iter in cluster_train_sample_indices.values()]
                    self.X_test = [X_test[x_iter] for x_iter in cluster_test_sample_indices.values()]
                    self.y_test = [y_test[x_iter] for x_iter in cluster_test_sample_indices.values()]
                else:
                    top_k_indices = torch.argsort(attention_score)[:, -min(train_len, X_train.shape[0]):]
                    cluster_num = min(cluster_num, len(top_k_indices))
                    cluster_train_sample_indices, cluster_test_sample_indices = cluster_test_data(top_k_indices,
                                                                                                  cluster_num)
                    self.X_train = [X_train[x_iter] for x_iter in cluster_train_sample_indices.values()]
                    self.y_train = [y_train[x_iter] for x_iter in cluster_train_sample_indices.values()]
                    self.X_test = [X_test[x_iter] for x_iter in cluster_test_sample_indices.values()]
                    self.y_test = [y_test[x_iter] for x_iter in cluster_test_sample_indices.values()]
            else:
                pass
                #TODO jianshengli Realize fine-tuning for each sample separately


