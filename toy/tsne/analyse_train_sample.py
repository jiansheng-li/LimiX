import sys
sys.path.insert(0,'/mnt/public/jianshengli/LimiX-extension')
from inference import InferenceAttentionMap, SubSampleData
from utils import find_top_K_indice, cluster_test_data




import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import make_blobs

import warnings
warnings.filterwarnings("ignore")
from utils.loading import load_model
from baseline.MLP import TwoLayerMLP, train_model, test_model, train_and_test_model
import pandas as pd
from tqdm import tqdm
import numpy as np
import gc
import torch
import argparse
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import roc_auc_score

from utils.inference_utils import  simple_inference

os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
from utils.utils import download_datset, download_model

if not torch.cuda.is_available():
    raise SystemError('GPU device not found. For fast training, please enable GPU.')


def auc_metric(target, pred, multi_class='ovo', numpy=False):
    lib = np if numpy else torch
    try:
        if not numpy:
            target = torch.tensor(target) if not torch.is_tensor(target) else target
            pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
        if len(lib.unique(target)) > 2:
            if not numpy:
                return torch.tensor(roc_auc_score(target, pred, multi_class=multi_class))
            return roc_auc_score(target, pred, multi_class=multi_class)
        else:
            if len(pred.shape) == 2:
                pred = pred[:, 1]
            if not numpy:
                return torch.tensor(roc_auc_score(target, pred))
            return roc_auc_score(target, pred)
    except ValueError as e:
        print(e)
        return np.nan if numpy else torch.tensor(np.nan)


# --- ECE (Expected Calibration Error) ---
def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error (ECE) implementation"""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    if y_prob.ndim == 2 and y_prob.shape[1] > 1:
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
    else:
        confidences = y_prob if y_prob.ndim == 1 else y_prob[:, 1]
        predictions = (confidences >= 0.5).astype(int)

    accuracies = (predictions == y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            avg_conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(acc_in_bin - avg_conf_in_bin) * prop_in_bin
    return ece


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run LimiX inference')
    parser.add_argument('--data_dir', type=str, default="/mnt/public/classifier_benchmarks/tabzilla",
                        help='Specify the local storage directory of the dataset')
    parser.add_argument('--save_name', default=None, type=str, help="path to save result")
    parser.add_argument('--debug', default=False, action='store_true', help="debug mode")
    parser.add_argument('--model_path', type=str, default="/mnt/public/jianshengli/Limix/LimiX-16M.ckpt",
                        help="path to you model")
    args = parser.parse_args()

    model_path = args.model_path
    data_root = args.data_dir


    scaler=MinMaxScaler()
    le=LabelEncoder()
    rsts = []
    # Iterate through all datasets and perform inference
    for idx, folder in tqdm(enumerate(os.listdir(data_root)),total=len(os.listdir(data_root)), desc=f"dataset",
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [耗时:{elapsed}]",leave=False):
        X_train, X_test, y_train, y_test = None, None, None, None
        folder_path = os.path.join(data_root, folder)
        if os.path.isfile(folder_path):
            continue

        try:
            # start_time_pre = time.time()
            train_path = os.path.join(folder_path, folder + '_train.csv')
            test_path = os.path.join(folder_path, folder + '_test.csv')
            if os.path.exists(train_path):
                train_df = pd.read_csv(train_path)

                if os.path.exists(test_path):
                    test_df = pd.read_csv(test_path)
                else:
                    # If there is no test.csv, split train.csv into training and testing sets
                    train_df, test_df = train_test_split(train_df, test_size=0.5, random_state=42)

            dataset_name = folder  # Use the folder name as the dataset name.

            # The last column is the target variable
            X_train = train_df.iloc[:, :-1]
            y_train = train_df.iloc[:, -1]
            X_test = test_df.iloc[:, :-1]
            y_test = test_df.iloc[:, -1]

            for col in X_train.columns:
                if X_train[col].dtype == 'object':  # Check whether it is a string column.
                    try:
                        le = LabelEncoder()
                        X_train[col] = le.fit_transform(X_train[col])
                        X_test[col] = le.transform(X_test[col])
                    except Exception as e:
                        X_train = X_train.drop(columns=[col])
                        X_test = X_test.drop(columns=[col])
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)
            num_classes = len(le.classes_)

            trainX, trainy = X_train, y_train

            trainX = np.asarray(trainX, dtype=np.float32)
            trainy = np.asarray(trainy, dtype=np.int64)

            # Datasets with too many or too few categories are not supported yet
            if len(np.unique(trainy)) > 10 or len(np.unique(trainy)) < 2:
                continue

            testX, testy = X_test, y_test
            testX = np.asarray(testX, dtype=np.float32)
            testy = np.asarray(testy, dtype=np.int64)
            if len(trainX)>50000:continue

            rst = {
                'dataset name': folder,
                'num_data_train': len(trainX),
                'num_data_test': len(testX),
                'num_feat': len(trainX[0]),
                'num_class': len(np.unique(trainy)),
            }

            model=load_model(model_path)

            save_path=f"{args.save_name}"
            os.makedirs(save_path,exist_ok=True)

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                      '#bcbd22', '#17becf']
            color_map = {label: colors[i] for i, label in enumerate(np.unique(trainy))}
            train_marker = 'o'  # 选中的点
            test_marker = '^'  # 未选中的点
            select_edge="black"
            unselect_edge="white"
            for layer in range(12):
                for sublayer in [1,3,5]:
                    embedding = simple_inference(model, trainX, trainy, trainX, "emb",return_layer_idx=layer,return_sublayer_idx=sublayer).clone().detach().cpu().numpy()
                    tsne = TSNE(n_components=2,
                                perplexity=min(30, len(trainy) // 2),
                                random_state=42,
                                init='pca',
                                learning_rate='auto',
                                n_iter=1000)
                    X_tsne = tsne.fit_transform(embedding[:len(trainy)])
                    fig, ax = plt.subplots(figsize=(10, 8))
                    for label in np.unique(trainy):
                        mask = trainy == label
                        indices_for_label = np.arange(len(trainy))[mask]
                        if len(indices_for_label) > 0:
                            ax.scatter(X_tsne[:, 0][indices_for_label],
                                       X_tsne[:, 1][indices_for_label],
                                       c=[color_map[label]],
                                       s=30,
                                       alpha=0.9,
                                       marker=train_marker,
                                       edgecolors=select_edge,
                                       linewidth=0.8,
                                       label=f'Selected Train Label {label}')
                    ax.legend(loc='upper right', fontsize=8, ncol=2)
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_path, f"{folder}_train_sublayer_{sublayer}_layer_{layer}.png"))
                    plt.close()
                    fig, ax = plt.subplots(figsize=(10, 8))
                    tsne = TSNE(n_components=2,
                                perplexity=min(30, len(testy) // 2),
                                random_state=42,
                                init='pca',
                                learning_rate='auto',
                                n_iter=1000)
                    X_tsne = tsne.fit_transform(embedding[len(trainy):])
                    for label in np.unique(trainy):
                        mask = trainy == label
                        indices_for_label = np.arange(len(trainy))[mask]
                        if len(indices_for_label) > 0:
                            ax.scatter(X_tsne[:, 0][indices_for_label],
                                       X_tsne[:, 1][indices_for_label],
                                       c=[color_map[label]],
                                       s=30,
                                       alpha=0.9,
                                       marker=test_marker,
                                       edgecolors=select_edge,
                                       linewidth=0.8,
                                       label=f'Selected Test Label {label}')
                    ax.legend(loc='upper right', fontsize=8, ncol=2)
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_path, f"{folder}_test_sublayer_{sublayer}_layer_{layer}.png"))
                    plt.close()



            # for i in range(20):
            #     fig, ax = plt.subplots(figsize=(10, 8))
            #     train_indices = cluster_train_sample_indices[i]
            #     test_indices = cluster_test_sample_indices[i]
            #     train_remain_indices = np.setdiff1d(np.arange(len(trainX)), train_indices)
            #     test_remain_indices = np.setdiff1d(np.arange(len(trainX)), test_indices)
            #     tsne = TSNE(n_components=2,
            #                 perplexity=min(30, len(train_indices) // 2),
            #                 random_state=42,
            #                 init='pca',
            #                 learning_rate='auto',
            #                 n_iter=1000)
            #     X_tsne = tsne.fit_transform(embedding[train_indices])
            #     colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
            #               '#bcbd22', '#17becf']
            #     color_map = {label: colors[i] for i, label in enumerate(np.unique(trainy))}

                # for label in np.unique(trainy):
                #     mask = trainy[train_remain_indices] == label
                #     indices_for_label = np.array(train_remain_indices)[mask]
                #     if len(indices_for_label) > 0:
                #         ax.scatter(X_tsne[:len(trainy), 0][indices_for_label],
                #                    X_tsne[:len(trainy), 1][indices_for_label],
                #                    c=[color_map[label]],
                #                    s=30,
                #                    alpha=0.6,
                #                    edgecolors=unselect_edge,
                #                    linewidth=0.8,
                #                    marker=train_marker,
                #                    label=f'Unselected Train Label {label}')


                # for label in np.unique(trainy):
                #     mask = testy[test_remain_indices] == label
                #     indices_for_label = np.array(test_remain_indices)[mask]
                #     if len(indices_for_label) > 0:
                #         ax.scatter(X_tsne[len(trainy):, 0][indices_for_label],
                #                    X_tsne[len(trainy):, 1][indices_for_label],
                #                    c=[color_map[label]],
                #                    s=30,
                #                    alpha=0.6,
                #                    edgecolors=unselect_edge,
                #                    linewidth=0.8,
                #                    marker=test_marker,
                #                    label=f'Unselected Test Label {label}')

                # for label in np.unique(trainy):
                #     mask = trainy[train_indices] == label
                #     indices_for_label = np.arange(len(train_indices))[mask]
                #     if len(indices_for_label) > 0:
                #         ax.scatter(X_tsne[:, 0],
                #                    X_tsne[:, 1],
                #                    c=[color_map[label]],
                #                    s=30,
                #                    alpha=0.9,
                #                    marker=train_marker,
                #                    edgecolors=select_edge,
                #                    linewidth=0.8,
                #                    label=f'Selected Train Label {label}')
                # ax.legend(loc='upper right', fontsize=8, ncol=2)
                # plt.tight_layout()
                # plt.savefig(os.path.join(save_path, f"{folder}_{i}_train.png"))
                # plt.close()
                # fig, ax = plt.subplots(figsize=(10, 8))
                # tsne = TSNE(n_components=2,
                #             perplexity=min(30, len(test_indices) // 2),
                #             random_state=42,
                #             init='pca',
                #             learning_rate='auto',
                #             n_iter=1000)
                # X_tsne = tsne.fit_transform(embedding[test_indices])
                # for label in np.unique(trainy):
                #     mask = trainy[test_indices] == label
                #     indices_for_label = np.arange(len(test_indices))[mask]
                #     if len(indices_for_label) > 0:
                #         ax.scatter(X_tsne[:, 0],
                #                    X_tsne[:, 1],
                #                    c=[color_map[label]],
                #                    s=30,
                #                    alpha=0.9,
                #                    marker=test_marker,
                #                    edgecolors=select_edge,
                #                    linewidth=0.8,
                #                    label=f'Selected Test Label {label}')
                # ax.legend(loc='upper right', fontsize=8, ncol=2)
                # plt.tight_layout()
                # plt.savefig(os.path.join(save_path, f"{folder}_{i}_test.png"))
                # plt.close()
        except Exception as e:
            print(f"Error processing: {e}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()









