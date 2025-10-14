import os
import pickle
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import pandas as pd
from tqdm import tqdm
import numpy as np
import gc
import torch
import argparse  # 新增参数解析模块
import warnings

from MICP.LoCalPFN import find_k_nearest_neighbors
from inference.inference_method import InferenceAttentionMap, setup, cleanup
from generate_node import get_X_y, plot_points
from inference.preprocess import SubSampleData
from utils.loading import load_model
from utils.retrieval_utils import find_top_K_indice

warnings.filterwarnings("ignore", category=UserWarning)

# 在参数解析部分添加新的参数
if __name__ == '__main__':
    model_path = "/mnt/public/jianshengli/Limix/LimiX-16M.ckpt"
    distribution_type = "iid"
    for i in range(10, 11):
        model = load_model(model_path).to("cuda")
        total_points = 8000
        trainX, trainy = get_X_y(i, total_points=total_points, data_type="train", distribution=distribution_type)
        testX, testy = get_X_y(i, total_points=i, data_type="test", distribution=distribution_type)


        # trainX,testX,trainy,testy=train_test_split(X,y,test_size=0.2,shuffle=True,stratify=y)
        # classifier = TabPFNClassifier(device="cuda", ignore_pretraining_limits=True,
        #                               model_path="/mnt/public/auto_inference/seed_y136_r190_m2260_l28_0630_1525/prior_diff_real_checkpoint_n_0_epoch_660-mfp.cpkt",
        #                               feature_select_type="quantile_uni_fine",
        #                               sample_selection_type="DP", calculate_sample_attention_score=False)
        # classifier.fit(trainX, trainy)
        # prediction_ = classifier.predict_proba(testX)
        inference_attention = InferenceAttentionMap(model_path=model_path, calculate_feature_attention=False,
                                                    calculate_sample_attention=True)
        feature_attention_score, sample_attention_score = inference_attention.inference_on_single_gpu(trainX, trainy, testX)
        calculate_attention_score = SubSampleData("sample", "only_sample")
        calculate_attention_score.fit(feature_attention_score=feature_attention_score,
                                      sample_attention_score=sample_attention_score, )
        sample_attention = calculate_attention_score.transform()
        # attention_model=load_model(path=model_path,calculate_sample_attention=True)[0].to("cuda")
        X_full = torch.cat([torch.tensor(trainX, dtype=torch.float32), torch.tensor(testX, dtype=torch.float32)],
                           dim=0).unsqueeze(0).to("cuda:0")
        # with (
        #     torch.autocast(torch.device("cuda").type, enabled=True),
        #     torch.inference_mode(),
        # ):
        #     output,feature_attention,sample_attention = attention_model(
        #         (None, X_full, torch.tensor(trainy, dtype=torch.float32).to(device="cuda"),0),
        #         only_return_standard_out=True,
        #         categorical_inds=[],
        #         single_eval_pos=trainy.shape[0],
        #     )

            # with (
            #     torch.autocast(torch.device("cuda").type, enabled=True),
            #     torch.inference_mode(),
            # ):
            #     output= model(x=X_full, y=torch.tensor(trainy).to("cuda:0").squeeze().unsqueeze(0),
            #                     eval_pos=torch.tensor(trainy).squeeze().unsqueeze(0).shape[1], task_type="reg")
            # output = np.argmax(output.squeeze().cpu().numpy(), axis=1)

            # top_k_indices = np.argsort(sample_attention[:, :].cpu())[:, -min(total_points // i, trainX.shape[0]):]
        top_k_indices = np.argsort(torch.mean(sample_attention_score,dim=0).cpu())[:, -min(800, trainX.shape[0]):]
        distance,knn_indices = find_k_nearest_neighbors(torch.tensor(trainX, dtype=torch.float32), torch.tensor(testX, dtype=torch.float32), 800)
        # top_k_indices = find_top_K_indice(sample_attention, threshold=0.9, mixed_method="max")
        for label in range(0,len(top_k_indices),len(top_k_indices)//i):
            sample_index=top_k_indices[label].cpu()
            k_index=knn_indices[label].cpu()
            plot_points(trainX[sample_index], trainy[sample_index], dir=f"select_sample_{distribution_type}",
                        data_type=f"top_k",
                        file_name=f"{len(np.unique(testy))}_label_{label}", test_X=testX[label], test_y=label)
            plot_points(trainX, trainy, dir=f"select_sample_{distribution_type}",
                        data_type=f"total",
                        file_name=f"{len(np.unique(testy))}_label_{label}", test_X=testX[label], test_y=label)
            plot_points(trainX[k_index], trainy[k_index], dir=f"select_sample_{distribution_type}",
                        data_type=f"knn",
                        file_name=f"{len(np.unique(testy))}_label_{label}", test_X=testX[label], test_y=label)
