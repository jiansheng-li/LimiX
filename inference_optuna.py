import sys

from inference.predictor import LimiXPredictor

sys.path.insert(0,'/mnt/public/jianshengli/LimiX')
print(sys.path)
import json
import os

import random
import time
import warnings
warnings.filterwarnings("ignore")
from inference.inference_method import InferenceAttentionMap, InferenceResultWithRetrieval
from inference.preprocess import SubSampleData
from retrieval_extension.retrieval_search_space.inference_search import RetrievalSearchHyperparameters
from utils.loading import load_model

import pandas as pd
from tqdm import tqdm
import numpy as np
import gc
import torch
import argparse
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.metrics import roc_auc_score
from pathlib import Path
import torch.distributed as dist
from utils.inference_utils import generate_infenerce_config

os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
from utils.utils import download_datset, download_model

if not torch.cuda.is_available():
    raise SystemError('GPU device not found. For fast training, please enable GPU.')


def process_single_folder_cls(folder_path, folder, model_path, args,idx):
    """
    处理单个文件夹的函数
    """
    scaler = StandardScaler()
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(idx%8)
    torch.cuda.set_device(idx%torch.cuda.device_count())
    X_train, X_test, y_train, y_test = None, None, None, None
    with open(args.inference_config_path, "r") as f:
        inference_config = json.load(f)
    if os.path.isfile(folder_path):
        return None

    try:
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

            # 为标签编码单独创建一个LabelEncoder
            label_le = LabelEncoder()
            y_train = label_le.fit_transform(y_train)
            y_test = label_le.transform(y_test)

            trainX, trainy = X_train, y_train

            trainX = np.asarray(trainX, dtype=np.float32)
            trainy = np.asarray(trainy, dtype=np.int32)

            # Datasets with too many or too few categories are not supported yet
            if len(np.unique(trainy)) > 10 or len(np.unique(trainy)) < 2:
                return None

            testX, testy = X_test, y_test
            testX = np.asarray(testX, dtype=np.float32)
            testy = np.asarray(testy, dtype=np.int32)

            rst = {
                'dataset name': folder,
                'num_data_train': len(trainX),
                'num_data_test': len(testX),
                'num_feat': len(trainX[0]),
                'num_class': len(np.unique(trainy)),
            }
            if len(trainX)>50000:
                return rst
            if len(trainX)<2000:
                args.n_trials=50
            searchInference = RetrievalSearchHyperparameters(
                dict(use_cluster=True,device_id=idx%torch.cuda.device_count(),model_path=model_path), trainX, trainy, testX, testy,
            )
            config, result = searchInference.search( n_trials=args.n_trials, metric="AUC",
                                                    inference_config=inference_config,task_type="cls")
            for i in range(len(inference_config)):
                inference_config[i]["retrieval_config"].update(config)
            classifier = LimiXPredictor(device=torch.device(
                "cuda:" + str(idx % torch.cuda.device_count()) if torch.cuda.is_available() else "cpu"),
                                        model_path=model_path,
                                        inference_config=inference_config)
            prediction_ = classifier.predict(trainX, trainy, testX, task_type="Classification")
            prediction_label = np.argmax(prediction_, axis=1)
            rst["2M-auc"] = result
            rst["2M-acc"] = float(accuracy_score(testy, prediction_label))
            rst["2M-f1"] = float(f1_score(testy, prediction_label, average='macro'))
            return rst
    except Exception as e:
        print(f"Error processing folder {folder}: {str(e)}")
        return None


def process_single_folder_reg(folder_path, folder, model_path, args,idx):

    scaler = StandardScaler()
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(idx%torch.cuda.device_count())
    torch.cuda.set_device(idx%torch.cuda.device_count())
    with open(args.inference_config_path, "r") as f:
        inference_config = json.load(f)

    try:
        train_data_path = Path(folder_path, f'{folder}_train.csv')
        test_data_path = Path(folder_path, f'{folder}_test.csv')
        trainX, trainy = train_data = load_data(train_data_path)
        testX, testy = test_data = load_data(test_data_path)
        rst = {
            'dataset name': folder_path,
            'num_data_train': len(trainX),
            'num_data_test': len(testX),
            'num_feat': trainX.shape[1],
            'num_class': len(np.unique(trainy)),
        }
        if len(trainX) > 50000:
            return rst
        searchInference = RetrievalSearchHyperparameters(
            dict(use_cluster=True,device_id=idx%torch.cuda.device_count(),model_path=model_path), trainX, trainy, testX, testy,
            )
        config, result = searchInference.search(n_trials=args.n_trials, metric="R2",inference_config=inference_config,task_type="Regression")
        for i in range(len(inference_config)):
            inference_config[i]["retrieval_config"].update(config)
        classifier = LimiXPredictor(device=torch.device("cuda:"+str(idx%torch.cuda.device_count()) if torch.cuda.is_available() else "cpu"),
                                    model_path=model_path,
                                    inference_config=inference_config)
        y_mean = trainy.mean()
        y_std = trainy.std()
        y_train_normalized = (trainy - y_mean) / y_std
        y_test_normalized = (testy - y_mean) / y_std
        y_pred = classifier.predict(trainX, y_train_normalized, testX, task_type="Regression")
        y_pred = y_pred.to('cpu')
        rmse = float(mean_squared_error(y_test_normalized, y_pred))
        rst["2M-r2"] = result
        rst["2M-rmse_normed"] = rmse
        return rst
    except Exception as e:
        # raise e
        print(f"Error processing {folder_path}: {e}")




def load_data(data_path):
    data = pd.read_csv(data_path)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].astype(np.float32)
    return X, y

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run LimiX inference')
    parser.add_argument('--data_dir', type=str, default="/mnt/public/regression_benchmarks/tabarena_reg",
                        help='Specify the local storage directory of the dataset')
    parser.add_argument('--save_name', default="./result/tabarena_regtest", type=str, help="path to save result")
    parser.add_argument('--inference_config_path', type=str, default="./config/reg_default_retrieval.json",
                        help="path to example config")
    parser.add_argument('--model_path', type=str, default="/mnt/public/jianshengli/Limix/LimiX-16M.ckpt",
                        help="path to you model")
    parser.add_argument('--debug', default=False, action='store_true', help="debug mode")
    parser.add_argument("--n_trials", default=2, type=int, help="n_trials")
    parser.add_argument("--use_cluster", default=True, action='store_false', help="use cluster")
    parser.add_argument("--task_type", default="reg", type=str, help="task type")
    args = parser.parse_args()

    model_path = args.model_path
    data_root = args.data_dir
    if data_root is None:
        download_datset(repo_id="stableai-org/bcco_cls", revision="main", save_dir="./cache")
        data_root = "./cache/bcco_cls"
    if model_path is None:
        model_path = download_model(repo_id="stableai-org/LimiX-16M", filename="LimiX-16M.ckpt", save_path="./cache")

    if args.save_name is None:
        # Dynamically generate the save path
        args.save_name = time.strftime("%Y%m%d-%H%M%S")
    if not args.debug:
        os.makedirs(args.save_name, exist_ok=True)
        save_result_path=os.path.join(args.save_name, f"rsts_onlysample_{args.n_trials}.csv")

    scaler=MinMaxScaler()
    le=LabelEncoder()
    rsts = []
    # Iterate through all datasets and perform inference
    folders = []
    for item in os.listdir(data_root):
        item_path = os.path.join(data_root, item)
        if os.path.isdir(item_path):
            folders.append((item_path, item, model_path, args))
    results = []
    import multiprocessing as mp

    mp.set_start_method('spawn', force=True)
    with mp.Pool(processes=torch.cuda.device_count()*4) as pool:
        async_results = []
        for idx,(folder_path, folder, model_path, args) in enumerate(folders):
            async_result = pool.apply_async(process_single_folder_reg if args.task_type=="reg" else process_single_folder_cls,
                                            args=(folder_path, folder, model_path, args,idx,))
            async_results.append(async_result)


        for async_result in tqdm(async_results, desc="Processing datasets",
                                 bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [耗时:{elapsed}]",
                                 leave=False):
            result = async_result.get()
            if result is not None:
                results.append(result)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    rstsdf = pd.DataFrame(results)
    if not args.debug:
        rstsdf.to_csv(save_result_path, index=False)





