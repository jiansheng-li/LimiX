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


def split_data(
        features,
        labels,
        max_chunk_size: int,
        *,
        equal_split_size: bool,
) -> tuple[list, list]:
    """Split a large dataset into chunks along the first dimension.

    Args:
        features: features array-like object
        labels: labels array-like object
        max_chunk_size: maximum size of each chunk. The function will choose
            the minimum number of chunks that keeps each chunk under this size.
        equal_split_size: If True, splits data into equally sized chunks
            (or as equal as possible under max_chunk_size).
            If False, splits into chunks of size `max_chunk_size`, with the
            last chunk having the remainder samples but is dropped if its
            size is less than 2.

    Returns:
        A tuple of two lists: one containing the feature chunks and one
        containing the corresponding label chunks.
    """
    total_size = len(features)

    # Validate input parameters
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be positive")
    if total_size == 0:
        return [], []

    if not equal_split_size:
        # Split into chunks of exactly max_chunk_size (except possibly the last one)
        min_batch_size = 2
        feature_chunks = []
        label_chunks = []

        # Process full chunks
        start_idx = 0
        while start_idx < total_size:
            end_idx = min(start_idx + max_chunk_size, total_size)
            current_chunk_size = end_idx - start_idx

            # Only add the chunk if it meets the minimum size requirement
            if current_chunk_size >= min_batch_size:
                feature_chunks.append(features[start_idx:end_idx])
                label_chunks.append(labels[start_idx:end_idx])
            # If the remaining chunk is smaller than min_batch_size, it's dropped

            start_idx = end_idx

        return feature_chunks, label_chunks

    # Calculate the number of chunks needed to stay under max_chunk_size
    num_chunks = (total_size + max_chunk_size - 1) // max_chunk_size  # Ceiling division

    # Calculate base chunk size and remainder
    base_chunk_size = total_size // num_chunks
    remainder = total_size % num_chunks

    # Create chunks with size as equal as possible
    feature_chunks = []
    label_chunks = []
    start_idx = 0

    for i in range(num_chunks):
        # Distribute the remainder among the first few chunks
        current_chunk_size = base_chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk_size

        feature_chunks.append(features[start_idx:end_idx])
        label_chunks.append(labels[start_idx:end_idx])

        start_idx = end_idx

    return feature_chunks, label_chunks


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
                 X_test: torch.Tensor=None,
                 y_test: torch.Tensor=None,
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
                     X_test: torch.Tensor=None,
                     y_test: torch.Tensor=None,
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
                trainX, trainy = split_data(X_train, y_train, train_len, equal_split_size=True)
                self.X_train=[]
                self.y_train=[]
                self.X_test=[]
                self.y_test=[]
                for X,y in zip(trainX,trainy):
                    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
                    self.X_train.append(X_train)
                    self.y_train.append(y_train)
                    self.X_test.append(X_test)
                    self.y_test.append(y_test)
                pass
                #TODO jianshengli Realize fine-tuning for each sample separately
        self.max_steps=len(self.X_train)
if __name__=="__main__":
    trainX=torch.randn(100000,10)
    trainy=torch.randint(0,2,(100000,))
    dataset=TabularFinetuneDataset(trainX,trainy)
    split_data(trainX,trainy,2000,equal_split_size=True)
