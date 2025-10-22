import os
import sys
sys.path.insert(0,'/mnt/public/jianshengli/LimiX-extension')
import argparse
import networkx as nx
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm

from inference.inference_method import InferenceAttentionMap, setup
from utils.loading import load_model

# 在参数解析部分添加新的参数
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run LimiX inference')
    parser.add_argument('--data_dir', type=str, default="/mnt/public/jianshengli/hn",
                        help='Specify the local storage directory of the dataset')
    parser.add_argument('--save_name', default=None, type=str, help="path to save result")
    parser.add_argument('--debug', default=False, action='store_true', help="debug mode")
    parser.add_argument('--model_path', type=str, default="/mnt/public/jianshengli/Limix/LimiX-16M.ckpt",
                        help="path to you model")
    args = parser.parse_args()

    model_path = args.model_path
    data_root = args.data_dir
    model_path = "/mnt/public/jianshengli/Limix/LimiX-16M.ckpt"
    model = load_model(model_path=model_path).to("cuda")
    scaler=MinMaxScaler()
    model.to("cuda")

    for idx, folder in tqdm(enumerate(os.listdir(data_root)),total=len(os.listdir(data_root)), desc=f"dataset",
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [耗时:{elapsed}]",leave=False):
        X_train, X_test, y_train, y_test = None, None, None, None
        folder_path = os.path.join(data_root, folder)
        if os.path.isfile(folder_path):
            continue


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

        inference_attention = InferenceAttentionMap(model_path=model_path, calculate_feature_attention=True,
                                                    calculate_sample_attention=True)
        feature_attention_score, sample_attention_score = inference_attention.inference_on_single_gpu(trainX, trainy, testX)
        plt.figure(figsize=(8, 4.2))
        feature_attention_score=feature_attention_score[:,-1,:]
        heatmap = sns.heatmap(feature_attention_score.cpu(), annot=False, fmt=".2f", cmap='YlGnBu',
                              cbar_kws={'label': 'Value'})
        plt.savefig(f"demo_showcase.pdf",format="pdf",dpi=450,bbox_inches='tight', pad_inches=0)
        plt.close()
    #

