import gc
import time
from typing import Literal, Tuple
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

from tqdm import tqdm

from utils.data_utils import TabularInferenceDataset, cluster_test_data
from utils.inference_utils import NonPaddingDistributedSampler, swap_rows_back
from utils.loading import load_model

from utils.retrieval_utils import RelabelRetrievalY, find_top_K_indice, find_top_K_class
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup():
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def cleanup():
    if not dist.is_initialized():
        print("Distributed environment is not initialized, nothing to clean up.")
        return

    print("Cleaning up distributed environment...")
    dist.destroy_process_group()
    print("Distributed environment cleaned up.")


class InferenceResultWithRetrieval:
    def __init__(self,
                 model: torch.nn.Module | str,
                 sample_selection_type: Literal["AM", "DDP"] = "AM",
                 ):
        if isinstance(model, str):
            model = load_model(model)
        self.model = model
        self.sample_selection_type = sample_selection_type
        self.dataset = None

    def _prepare_data(self,
                      X_train: torch.Tensor,
                      y_train: torch.Tensor,
                      X_test: torch.Tensor,
                      attention_score: np.ndarray = None,
                      retrieval_len: int = 2000,
                      use_cluster: bool = False,
                      cluster_num: int = None,
                      use_threshold: bool = False,
                      mixed_method: str = "max",
                      threshold: float = None
                      ) -> TabularInferenceDataset:
        if self.sample_selection_type == "AM":
            use_retrieval = True
        else:
            use_retrieval = False
        dataset = TabularInferenceDataset(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            attention_score=attention_score,
            retrieval_len=retrieval_len,
            use_retrieval=use_retrieval,
            cluster_num=cluster_num,
            use_cluster=use_cluster,
            use_threshold=use_threshold,
            mixed_method=mixed_method,
            threshold=threshold
        )
        return dataset

    def inference(self,
                  X_train: torch.Tensor | np.ndarray = None,
                  y_train: torch.Tensor | np.ndarray = None,
                  X_test: torch.Tensor | np.ndarray = None,
                  dataset: TabularInferenceDataset = None,
                  attention_score: np.ndarray | torch.Tensor = None,
                  retrieval_len: int = 2000,
                  dynamic_ratio: float = None,
                  use_cluster: bool = False,
                  cluster_num: int | str = 2,
                  task_type: Literal["reg", "cls"] = "cls",
                  cluster_method: str = "overlap",
                  use_threshold: bool = False,
                  threshold: float = 0.5,
                  mixed_method: str = "max",device_id: int = 0
                  ):
        device = torch.device(f"cuda:{device_id}")
        if isinstance(X_train, np.ndarray):
            X_train = torch.from_numpy(X_train).float()
        if isinstance(y_train, np.ndarray):
            y_train = torch.from_numpy(y_train).float()
        if isinstance(X_test, np.ndarray):
            X_test = torch.from_numpy(X_test).float()

        if isinstance(retrieval_len, str):
            if retrieval_len == "dynamic":
                if dynamic_ratio is not None:
                    retrieval_len = int(dynamic_ratio * X_train.shape[0] / len(torch.unique(y_train)))
                else:
                    retrieval_len = int(X_train.shape[0] / len(torch.unique(y_train)))
        if isinstance(retrieval_len, float):
            self.retrieval_len = int(retrieval_len * X_train.shape[0])
        else:
            self.retrieval_len = retrieval_len

        if use_cluster:
            if cluster_num == "num_class":
                cluster_num = len(torch.unique(y_train))

            if use_threshold:
                top_k_indices = find_top_K_indice(attention_score, threshold=threshold, mixed_method=mixed_method,
                                                  retrieval_len=self.retrieval_len)
            else:
                # use retrieval len
                top_k_indices = torch.argsort(attention_score)[:, -min(self.retrieval_len, X_train.shape[0]):].to(
                    "cuda")

            cluster_num = min(cluster_num, len(top_k_indices))

            cluster_train_sample_indices, cluster_test_sample_indices = cluster_test_data(top_k_indices,
                                                                                          cluster_num,
                                                                                          cluster_method=cluster_method)
            X_train = [X_train[x_iter] for x_iter in cluster_train_sample_indices.values()]
            y_train = [y_train[x_iter] for x_iter in cluster_train_sample_indices.values()]
            X_test = [X_test[x_iter] for x_iter in cluster_test_sample_indices.values()]


            model = self.model.to(device)
            outputs = []
            total_indice = []
            for cluster_indices in cluster_test_sample_indices.values():
                for indices in cluster_indices:
                    total_indice.append(indices)
            for inference_idx in tqdm(range(len(cluster_test_sample_indices)), desc=f"inference",
                                      bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [耗时:{elapsed}]",
                                      leave=False):
                X_train_ = X_train[inference_idx]
                y_ = y_train[inference_idx].to(device).unsqueeze(0)
                X_test_ = X_test[inference_idx]
                x_ = torch.cat([X_train_, X_test_], dim=0).to(device).unsqueeze(0)
                if task_type == "cls":
                    relabel = RelabelRetrievalY(y_.unsqueeze(-1))
                    y_ = relabel.transform_y().squeeze(-1).to(device)
                with (
                    torch.autocast(device.type, enabled=True),
                    torch.inference_mode(),
                ):
                    output = model(x=x_, y=y_, eval_pos=y_.shape[1],
                                   task_type=task_type)
                if len(output.shape) == 3:
                    output = output.view(-1, output.shape[-1])
                if task_type == "cls":
                    output = output.cpu().numpy()
                    output = torch.cat(
                        [torch.from_numpy(relabel.inverse_transform_y(np.expand_dims(i, axis=0))) for i in
                         output], dim=0)
                outputs.append(output)
            outputs = torch.cat(outputs, dim=0)
            outputs = swap_rows_back(outputs, total_indice)
            return outputs
        else:

            if dataset is None:
                dataset = self._prepare_data(X_train, y_train, X_test, attention_score, self.retrieval_len)
            self.rank, self.world_size = setup()

            model = self.model.cuda(self.rank)
            model = DDP(model, device_ids=[self.rank], find_unused_parameters=False)
            sampler = NonPaddingDistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)
            outputs = []
            dataloader = DataLoader(dataset,
                                    batch_size=min(max(1, 000, 000 // self.retrieval_len // X_train.shape[1], 1),
                                                   8) if self.sample_selection_type == "AM" else 1024,
                                    shuffle=False,
                                    drop_last=False,
                                    sampler=sampler
                                    )
            indice = []
            for batch_idx, data in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"inference",
                                        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [耗时:{elapsed}]",
                                        leave=False) if self.rank == 0 else enumerate(
                dataloader):
                with (
                    torch.autocast(torch.device(model.device).type, enabled=True),
                    torch.inference_mode(),
                ):
                    if self.sample_selection_type == "DDP":
                        indice.append(data["idx"])
                        X_test = data["X_test"]
                        x_ = torch.cat([X_train, X_test], dim=0).unsqueeze(dim=0)

                        y_ = y_train.unsqueeze(0)

                        output = model(x=x_, y=y_, eval_pos=y_.shape[1],
                                       task_type=task_type)
                        output = output.permute(1, 0, 2)
                        if len(output.shape) == 3:
                            output = output.view(-1, output.shape[-1])
                    else:
                        indice.append(data["idx"])
                        X_train = data["X_train"]
                        X_test = data["X_test"].unsqueeze(1)
                        y_ = data["y_train"]
                        x_ = torch.cat([X_train, X_test], dim=1)
                        if task_type == "cls":
                            relabel = RelabelRetrievalY(y_)
                            y_ = relabel.transform_y().to(device)
                        output = model(x=x_, y=y_.squeeze(-1), eval_pos=y_.shape[1], task_type=task_type)
                        if len(output.shape) == 3:
                            output = output.view(-1, output.shape[-1])
                        if task_type == "cls":
                            output = output.cpu().numpy()
                            output = relabel.inverse_transform_y(output,
                                                                 num_classes=max(10, torch.unique(y_train).shape[0]))
                            output = torch.tensor(output, dtype=torch.float32, device=model.device)

                outputs.append(output.cpu())
                del output
                gc.collect()
                torch.cuda.empty_cache()
            del model
            outputs = torch.cat(outputs, dim=0)
            local_result_cpu = outputs.cpu()
            indice = torch.cat(indice, dim=0)
            local_indice_cpu = indice.cpu()
            outputs = [None for _ in range(self.world_size)]
            gathered_indice = [None for _ in range(self.world_size)]
            dist.all_gather_object(gathered_indice, local_indice_cpu)
            dist.all_gather_object(outputs, local_result_cpu)
            del local_result_cpu
            outputs = torch.cat(outputs, dim=0).to(torch.float32)
            gathered_indice = torch.cat(gathered_indice, dim=0)
            outputs = swap_rows_back(outputs, gathered_indice)
            gc.collect()
            torch.cuda.empty_cache()
            return outputs.squeeze(0)






class InferenceAttentionMap:
    def __init__(self,
                 model_path: str,
                 calculate_feature_attention: bool = False,
                 calculate_sample_attention: bool = False,
                 ):
        self.calculate_feature_attention = calculate_feature_attention
        self.calculate_sample_attention = calculate_sample_attention
        self.model = load_model(model_path, calculate_feature_attention=calculate_feature_attention,
                                calculate_sample_attention=calculate_sample_attention)

        self.dataset = None

    def _prepare_data(self,
                      X_train: torch.Tensor,
                      y_train: torch.Tensor,
                      X_test: torch.Tensor,
                      ) -> TabularInferenceDataset:
        dataset = TabularInferenceDataset(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            use_retrieval=False
        )
        return dataset

    def inference_on_multi_gpu(self,
                               X_train: torch.Tensor | np.ndarray,
                               y_train: torch.Tensor | np.ndarray,
                               X_test: torch.Tensor | np.ndarray,
                               task_type: Literal["reg", "cls"] = "reg") -> tuple[
        torch.Tensor | None, torch.Tensor | None]:
        self.rank, self.world_size = setup()
        # device = torch.device(f"cuda:{self.rank}")
        model = self.model.cuda(self.rank)
        model = DDP(model, device_ids=[self.rank])
        model.eval()
        if isinstance(X_train, np.ndarray):
            X_train = torch.from_numpy(X_train).float()
        if isinstance(y_train, np.ndarray):
            y_train = torch.from_numpy(y_train)
        if isinstance(X_test, np.ndarray):
            X_test = torch.from_numpy(X_test).float()
        dataset = self._prepare_data(X_train, y_train, X_test)

        sampler = NonPaddingDistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)
        dataloader = DataLoader(dataset,
                                batch_size=X_test.shape[0] // self.world_size,
                                shuffle=False,
                                drop_last=False,
                                sampler=sampler
                                )
        local_feature_attention = []
        local_sample_attention = []
        feature_attention = None
        sample_attention = None
        indice = []
        for batch_idx, data in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"inference attention map",
                                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [耗时:{elapsed}]",
                                    leave=False) if self.rank == 1 else enumerate(
            dataloader):
            X_test = data["X_test"]
            idx = data["idx"]
            indice.append(idx)
            x_ = torch.cat([X_train, X_test], dim=0).unsqueeze(dim=0)

            y_ = y_train.unsqueeze(0)
            with(torch.autocast('cuda', enabled=True), torch.inference_mode()):
                output, feature_attention, sample_attention = model(x=x_, y=y_, eval_pos=y_.shape[1],
                                                                    task_type=task_type)
            if self.calculate_sample_attention:
                local_sample_attention.append(sample_attention.permute(1, 0, 2))
            if self.calculate_feature_attention:
                local_feature_attention.append(feature_attention[y_.shape[1]:, :, :])
            del output, sample_attention, feature_attention, X_test
            gc.collect()
            torch.cuda.empty_cache()

        indice = torch.cat(indice, dim=0)
        if self.calculate_feature_attention:
            if self.world_size > 1:
                feature_attentions = torch.cat(local_feature_attention,
                                               dim=0)  # shape[len_Dtest, feature_num//feature_per_group, feature_num//feature_per_group,]
                local_result_cpu = feature_attentions.cpu()
                local_indice_cpu = indice.cpu()
                gathered_feature = [None for _ in range(self.world_size)]
                gathered_indice = [None for _ in range(self.world_size)]
                dist.all_gather_object(gathered_feature, local_result_cpu)
                dist.all_gather_object(gathered_indice, local_indice_cpu)
                feature_attention = torch.cat(gathered_feature, dim=0)
                gathered_indice = torch.cat(gathered_indice, dim=0)
                feature_attention = swap_rows_back(feature_attention, gathered_indice)
                del gathered_feature
        if self.calculate_sample_attention:
            if self.world_size > 1:
                sample_attentions = torch.cat(local_sample_attention, dim=0)
                local_indice_cpu = indice.cpu()
                local_result_cpu = sample_attentions.cpu()
                gathered_sample = [None for _ in range(self.world_size)]
                gathered_indice = [None for _ in range(self.world_size)]
                dist.all_gather_object(gathered_sample, local_result_cpu)
                dist.all_gather_object(gathered_indice, local_indice_cpu)
                sample_attention = torch.cat(gathered_sample, dim=0)
                gathered_indice = torch.cat(gathered_indice, dim=0)
                sample_attention = swap_rows_back(sample_attention, gathered_indice)
                del gathered_sample

        dist.barrier()
        del sample_attentions, feature_attentions, model
        gc.collect()
        torch.cuda.empty_cache()
        return feature_attention, sample_attention.permute(1, 0, 2)
        # self.rank, self.world_size = setup()
        # feature_attention=torch.empty(size=(X_test.shape[0],(X_test.shape[1]+1)//2+1,(X_test.shape[1]+1)//2+1), dtype=torch.float16, device="cpu")
        # sample_attention=torch.empty(size=((X_test.shape[1]+1)//2+1,X_test.shape[0],X_train.shape[0]), dtype=torch.float16, device="cpu")
        # if self.rank == 0:
        #     model = self.model.to("cuda")
        #     model.eval()
        #     if isinstance(X_train, np.ndarray):
        #         X_train = torch.from_numpy(X_train).float()
        #     if isinstance(y_train, np.ndarray):
        #         y_train = torch.from_numpy(y_train).float()
        #     if isinstance(X_test, np.ndarray):
        #         X_test = torch.from_numpy(X_test).float()
        #
        #     x_ = torch.cat([X_train, X_test], dim=0).unsqueeze(dim=0).to("cuda")
        #
        #     y_ = y_train.unsqueeze(0).to("cuda")
        #     with(torch.autocast('cuda', enabled=True), torch.inference_mode()):
        #         output, feature_attention, sample_attention = model(x=x_, y=y_, eval_pos=y_.shape[1],
        #                                                             task_type=task_type)
        #     gc.collect()
        #     torch.cuda.empty_cache()
        #     del model,x_,y_
        #     dist.broadcast(feature_attention, src=0)
        #     dist.broadcast(sample_attention, src=0)
        #     gc.collect()
        # dist.barrier()
        #
        # return feature_attention, sample_attention

    def inference_on_single_gpu(self,
                                X_train: torch.Tensor | np.ndarray,
                                y_train: torch.Tensor | np.ndarray,
                                X_test: torch.Tensor | np.ndarray,
                                task_type: Literal["reg", "cls"] = "cls",
                                device_id: int = 0) -> tuple[torch.Tensor | None, torch.Tensor | None]:

        device = torch.device(f"cuda:{device_id}")
        # device = torch.device(device_id)
        model = self.model.to(device)
        model.eval()
        if isinstance(X_train, np.ndarray):
            X_train = torch.from_numpy(X_train).float()
        if isinstance(y_train, np.ndarray):
            y_train = torch.from_numpy(y_train)
        if isinstance(X_test, np.ndarray):
            X_test = torch.from_numpy(X_test).float()

        with(torch.autocast(device.type, enabled=True), torch.inference_mode()):
            x_ = torch.cat([X_train, X_test], dim=0).unsqueeze(dim=0).to(device)
            y_ = y_train.unsqueeze(0).to(device)

            output, feature_attention, sample_attention = model(x=x_, y=y_, eval_pos=y_.shape[1],
                                                                task_type=task_type)
            gc.collect()
            torch.cuda.empty_cache()

        torch.cuda.empty_cache()
        return feature_attention[y_.shape[1]:] if feature_attention is not None else None, sample_attention if sample_attention is not None else None
