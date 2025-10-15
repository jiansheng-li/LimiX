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
    parser.add_argument('--data_dir', type=str, default="/mnt/public/classifier_benchmarks/tabarena_classification",
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
    # os.makedirs(args.save_name,exist_ok=True)
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
            if len(trainX) > 50000: continue
            rst = {
                'dataset name': folder,
                'num_data_train': len(trainX),
                'num_data_test': len(testX),
                'num_feat': len(trainX[0]),
                'num_class': len(np.unique(trainy)),
            }
            model = load_model(model_path)


            # origin_output=res["cls_output_ori_context"].squeeze(0)
            #
            # output = origin_output[:trainy.shape[0], :len(np.unique(trainy))].float()
            # outputs = torch.nn.functional.softmax(output, dim=1)
            #
            # output = outputs.float().cpu().numpy()
            # prediction_ = output / output.sum(axis=1, keepdims=True)
            # roc = auc_metric(trainy, prediction_)
            # print(f"original roc: {roc}")

            """
            base context inference
            """
            # res = simple_inference(model, trainX, trainy, testX, return_all_information=True)
            # transformer_ouput = res["cls_output_all"].squeeze(0)
            # output = transformer_ouput[:trainy.shape[0], :len(np.unique(trainy))].float()
            # outputs = torch.nn.functional.softmax(output, dim=1)
            #
            # output = outputs.float().cpu().numpy()
            # prediction_ = output / output.sum(axis=1, keepdims=True)
            # roc = auc_metric(trainy, prediction_)
            # print(f"transformer train roc: {roc}")
            # rst["context_AUC"]=float(roc)
            # transformer_ouput = res["cls_output_all"].squeeze(0)
            # output = transformer_ouput[trainy.shape[0]:, :len(np.unique(trainy))].float()
            # outputs = torch.nn.functional.softmax(output, dim=1)
            #
            # output = outputs.float().cpu().numpy()
            # prediction_ = output / output.sum(axis=1, keepdims=True)
            # roc = auc_metric(testy, prediction_)
            # print(f"transformer test roc: {roc}")
            # rst["test_AUC"]=float(roc)
            """
            retrieval context inference
            """



            rsts.append(rst)


        except Exception as e:
            print(f"Error processing: {e}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        result=pd.DataFrame(rsts)
        if not args.debug:
            result.to_csv(os.path.join(args.save_name,"rsts.csv"),index=False)







