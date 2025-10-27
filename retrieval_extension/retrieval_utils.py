import numpy as np
import torch
from typing import List

from sklearn.preprocessing import LabelEncoder


def cluster_test_data(top_k_indices: torch.Tensor | List[torch.Tensor], k_groups, cluster_method="overlap"):
    """
        Clusters test samples based on their top-k training data indices.
        Handles variable-length 1D tensors in a list.

        Args:
            top_k_indices (torch.Tensor or List[torch.Tensor]):
                - If Tensor: shape (n_test_samples, k) - fixed k.
                - If List[Tensor]: List of 1D tensors, shape (k_i,). k_i can vary.
                                   List length is n_test_samples.
            k_groups (int): The number of clusters.
            cluster_method (str): "overlap" for unique union, else raw indices.

        Returns:
            Tuple[clusters_unions, clusters_sample_indices]:
            - clusters_unions (dict): {cluster_id: union_indices_tensor}
            - clusters_sample_indices (dict): {cluster_id: indices_tensor}. These indices
                                              refer to the position in the original
                                              list/tensor of test samples.
        """
    is_list_input = False

    if isinstance(top_k_indices, list):
        is_list_input = True
        if not top_k_indices:
            raise ValueError("Input list 'top_k_indices' is empty.")
        if not all(isinstance(t, torch.Tensor) and t.dim() == 1 for t in top_k_indices):
            raise TypeError("All elements in the list 'top_k_indices' must be 1D torch.Tensors.")
        n_test_samples = len(top_k_indices)
        # max_train_len is the maximum length of the 1D tensors
        max_train_len = max(t.shape[0] for t in top_k_indices) if n_test_samples > 0 else 0
        # We don't need to create a unified 2D tensor; we process the list directly.
        # processed_top_k_indices conceptually remains the list itself for this path.

    else:  # It's a single 2D tensor
        if not isinstance(top_k_indices, torch.Tensor) or top_k_indices.dim() != 2:
            raise TypeError("Input 'top_k_indices' must be a 2D torch.Tensor or a List of 1D torch.Tensors.")
        top_k_indices = top_k_indices.to("cuda")
        n_test_samples = top_k_indices.shape[0]
        max_train_len = top_k_indices.shape[1]

    if is_list_input:
        # Flatten the list of tensors and find unique indices
        flattened_indices = torch.cat(top_k_indices, dim=0)
    else:
        flattened_indices = top_k_indices.flatten()

    unique_indices = torch.unique(flattened_indices)
    num_unique = len(unique_indices)

    index_to_col = {idx.item(): i for i, idx in enumerate(unique_indices)}

    # Build sparse binary matrix representation
    rows = []
    cols = []
    data_vals = []

    if is_list_input:
        # --- Process List[1D Tensors] ---
        for i, sample_indices_tensor in enumerate(top_k_indices):  # i is the test sample index
            k_i = sample_indices_tensor.shape[0]
            for j in range(k_i):  # j iterates through the indices of the current sample
                rows.append(i)
                # Map the actual training index to its column in the binary matrix
                cols.append(index_to_col[sample_indices_tensor[j].item()])
                data_vals.append(1)
    else:
        # --- Process single 2D Tensor ---
        k_fixed = top_k_indices.shape[1]
        for i in range(n_test_samples):
            for j in range(k_fixed):
                rows.append(i)
                cols.append(index_to_col[top_k_indices[i, j].item()])
                data_vals.append(1)

    # Create sparse tensor (assuming GPU usage as per original code)
    # Check if we have any data to create the tensor
    if not rows:
        # Handle edge case where there are no indices (e.g., empty input or all empty tensors)
        # Create an empty sparse tensor
        indices_sparse = torch.empty((2, 0), dtype=torch.long, device="cuda")
        values_sparse = torch.empty(0, dtype=torch.float, device="cuda")
        binary_matrix_sparse = torch.sparse_coo_tensor(
            indices_sparse, values_sparse, (n_test_samples, num_unique), device="cuda"
        )
    else:
        indices_sparse = torch.LongTensor([rows, cols]).to("cuda")
        values_sparse = torch.FloatTensor(data_vals).to("cuda")
        binary_matrix_sparse = torch.sparse_coo_tensor(
            indices_sparse, values_sparse, (n_test_samples, num_unique), device="cuda"
        )

    # --- Compute overlap matrix ---
    binary_dense = binary_matrix_sparse.to_dense()
    # Matrix multiplication to get pairwise overlaps (n_test_samples x n_test_samples)
    # This works even if the original "k" was variable, as the dense matrix captures the relationship.
    overlap_matrix = torch.mm(binary_dense, binary_dense.t())

    # Adjust diagonal: overlap of a sample with itself
    # For variable k, the self-overlap is the number of indices for that specific sample.
    diag_indices = torch.arange(n_test_samples, device="cuda")
    if is_list_input:
        # Diagonal entry (i,i) should be the count of indices for sample i
        diag_values = torch.tensor([len(t) for t in top_k_indices], dtype=torch.float, device="cuda")
    else:
        # For fixed k tensor, diagonal is simply k
        diag_values = torch.full((n_test_samples,), max_train_len, dtype=torch.float, device="cuda")

    overlap_matrix[diag_indices, diag_indices] = diag_values

    # --- Perform clustering ---
    # Assuming gpu_kmeans takes (data_points, k) and returns labels
    cluster_labels = gpu_kmeans(binary_dense, k_groups)

    # --- Prepare results ---
    clusters_sample_indices = {}
    clusters_unions = {}

    for cluster_id in range(k_groups):
        # Find sample indices belonging to this cluster
        indices_in_cluster = torch.where(cluster_labels == cluster_id)[0]

        if len(indices_in_cluster) > 0:
            if cluster_method == "overlap":
                # Take the unique union of training indices for samples in this cluster
                if is_list_input:
                    # Concatenate indices from all samples in the cluster, then take unique
                    indices_to_concat = [top_k_indices[i] for i in indices_in_cluster]
                    if indices_to_concat:  # Check if list is not empty
                        union_indices = torch.unique(torch.cat(indices_to_concat, dim=0))
                    else:
                        union_indices = torch.empty(0, dtype=torch.long, device="cuda")  # Or appropriate device
                else:
                    union_indices = torch.unique(top_k_indices[indices_in_cluster, :])

            else:  # cluster_method != "overlap"
                # Take all training indices (may contain duplicates)
                if is_list_input:
                    # Concatenate indices from all samples in the cluster
                    indices_to_concat = [top_k_indices[i] for i in indices_in_cluster]
                    if indices_to_concat:
                        union_indices = torch.cat(indices_to_concat, dim=0)
                    else:
                        union_indices = torch.empty(0, dtype=torch.long, device="cuda")
                else:
                    union_indices = top_k_indices[indices_in_cluster, :].flatten()

            clusters_unions[cluster_id] = union_indices.cpu()
            clusters_sample_indices[cluster_id] = indices_in_cluster.cpu()

    return clusters_unions, clusters_sample_indices


def gpu_kmeans(data, k, max_iters=100, tol=1e-4):
    """
        Kmeans method on gpu
    Args:
        data (torch.Tensor): The input data.
        k (int): The number of clusters.
        max_iters (int, optional): The maximum number of iterations. Defaults to 100.
        tol (float, optional): The tolerance for convergence. Defaults to 1e-4.
    Returns:
        torch.Tensor: The cluster labels.
    """
    n_samples = data.shape[0]
    # init centroids
    data_dense = data.to_dense() if data.is_sparse else data
    centroids = data_dense[torch.randperm(n_samples)[:k]]

    for _ in range(max_iters):
        # calculate distance
        distances = torch.cdist(data_dense, centroids)

        labels = torch.argmin(distances, dim=1)

        new_centroids = torch.stack([
            data_dense[labels == i].mean(dim=0) for i in range(k)
        ])

        if torch.norm(new_centroids - centroids) < tol:
            break

        centroids = new_centroids

    return labels


class RelabelRetrievalY:
    def __init__(self, y_train: torch.Tensor):
        """
        Args:
            y_train: (batch_size, n_samples, 1)
        """
        self.y_train = y_train.cpu().numpy()
        self.label_encoders = [LabelEncoder() for i in range(y_train.shape[0])]

    def transform_y(self, ):
        for i in range(self.y_train.shape[0]):
            self.y_train[i] = np.expand_dims(self.label_encoders[i].fit_transform(self.y_train[i].ravel()), axis=1)
        self.label_y = self.y_train.copy().astype(np.int32)
        self.y_train = torch.tensor(self.y_train, dtype=torch.float32)
        return self.y_train

    def inverse_transform_y(self, X: np.ndarray, num_classes: int = None) -> np.ndarray:
        """
        Args:
            X: (batch_size,  n_classes)
        Returns:
            X: (batch_size, n_classes)
        """
        if num_classes is None:
            num_classes = 10
        result = np.full((X.shape[0], num_classes), fill_value=-np.inf)
        for i in range(X.shape[0]):
            batch_label = np.unique(self.label_y[i])
            reverse_perm = self.label_encoders[i].inverse_transform(batch_label).astype(np.int32)
            reverse_output = np.full(num_classes, fill_value=-np.inf)
            reverse_output[reverse_perm] = X[i, batch_label]
            result[i] = reverse_output
        return result


def find_top_K_indice(sample_attention: torch.Tensor | np.ndarray, threshold: float = 0.5, mixed_method: str = "max",
                      retrieval_len: int = 200, device: str | torch.device = "cuda"):
    """
        Finds the indices of the largest elements in each row such that their sum
        is at least 90% of the total sum of that row.

        Args:
            tensor (torch.Tensor): A 2D tensor with values between 0 and 1.

        Returns:
            list[torch.Tensor[int]]: A list where each element is a list of column indices
                             for the corresponding row in the input tensor.
        """
    num_test_sample, num_train_sample = sample_attention.shape
    result_indices = []

    # 1. Calculate the target sum (90%) for each row
    row_sums = torch.sum(sample_attention, dim=1, keepdim=True)  # Shape: [num_rows, 1]
    target_sums = row_sums * threshold  # Shape: [num_rows, 1]

    for i in range(num_test_sample):
        # Get the current row
        current_row = sample_attention[i]  # Shape: [num_cols]

        # 2. Sort the row in descending order and get the indices
        # torch.sort returns (sorted_values, sorted_indices)
        sorted_values, sorted_indices = torch.sort(current_row, descending=True)

        # 3. Calculate cumulative sum of sorted values
        cumsum_values = torch.cumsum(sorted_values, dim=0)  # Shape: [num_cols]

        # 4. Find the first index where cumulative sum meets or exceeds the target
        # This finds indices where the condition is True, then we take the first one.
        # If the target sum is 0 (row of all zeros), this will select the first element (index 0).
        target_sum_for_row = target_sums[i].item()
        if target_sum_for_row > 0:
            # Use >= for robustness if cumsum exactly equals target
            sufficient_indices_mask = cumsum_values >= target_sum_for_row
            # .nonzero() returns indices of True elements. [0] gets the first one.
            # If no element meets the condition (unlikely with 90%), it might error.
            # We assume at least one element will eventually meet it due to sum >= target.
            try:
                num_elements_needed = torch.nonzero(sufficient_indices_mask, as_tuple=True)[0][0].item() + 1
                if mixed_method == "max":
                    num_elements_needed = min(max(num_elements_needed, retrieval_len), 10000)#max retrieval sample is 10000
                else:
                    num_elements_needed = min(num_elements_needed, retrieval_len)
            except IndexError:
                # Fallback: if no element meets the threshold (e.g., due to floating point),
                # select all elements.
                print(f"Warning: Row {i} might not reach 90% sum with individual elements. Selecting all.")
                num_elements_needed = num_train_sample
        else:
            # If target sum is 0, technically any set including the largest element works.
            # We select the largest element.
            num_elements_needed = 1

        # 5. Get the original column indices of the selected elements
        selected_original_indices = sorted_indices[:num_elements_needed].tolist()

        result_indices.append(torch.tensor(selected_original_indices, device=device))

    return result_indices


def find_top_K_class(X: torch.Tensor, num_class: int = 10):
    unique_classes, counts = torch.unique(X, return_counts=True)

    sorted_indices = torch.argsort(counts, descending=True)
    top_K_classes = unique_classes[sorted_indices[:num_class]]

    all_occurrence_indices = []
    for cls in top_K_classes:
        indices = (X == cls).nonzero(as_tuple=True)[0]
        all_occurrence_indices.append(indices)

    all_indices_flat = torch.cat(all_occurrence_indices) if all_occurrence_indices else torch.tensor([])
    return all_indices_flat