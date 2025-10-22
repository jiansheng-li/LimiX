import os
import sys



sys.path.insert(0,"/mnt/public/jianshengli/LimiX-extension/")
print(sys.path)
import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from toy.ER_causal_data_gen import generate_datasets
from inference.inference_method import InferenceAttentionMap, setup
from utils.loading import load_model
from utils.inference_utils import auc_metric, simple_inference

def duplicate_tensor_columns(tensor, n=None):
    """
    复制PyTorch tensor的每一列，在每一列后面紧跟着复制该列

    参数:
    tensor: PyTorch tensor (2D)
    n: 要复制的前n列，如果为None则复制所有列（除了最后一列）

    返回:
    复制后的tensor
    """
    if tensor.numel() == 0 or tensor.dim() != 2:
        return tensor

    num_cols = tensor.shape[1]




    # 提取需要复制的列
    cols_to_duplicate = tensor[:, :]



    # 创建复制列
    duplicated_cols = cols_to_duplicate.clone()  # 使用clone()复制tensor

    # 将原始列和复制列交替排列
    result_cols = []
    for i in range(num_cols):
        result_cols.append(cols_to_duplicate[:, i:i + 1])  # 原始列
        result_cols.append(duplicated_cols[:, i:i + 1])  # 复制列

    # 将处理后的列与最后一列合并
    if len(result_cols) > 0:
        duplicated_part = torch.cat(result_cols, dim=1)



    return duplicated_part
# 在参数解析部分添加新的参数
if __name__ == '__main__':
    model_path = "/mnt/public/jianshengli/Limix/LimiX-16M.ckpt"
    model = load_model(model_path=model_path).to("cuda")


    features_per_node=1
    datasets=generate_datasets(8, 10, 2000,n_datasets=1,features_per_node=features_per_node,function_type="MLP",noise_std=0.1)
    X_full=datasets[0][1].to("cuda").squeeze()
    y_full=datasets[0][2].to("cuda").squeeze()
    X_full=duplicate_tensor_columns(X_full)
    adjacency=datasets[0][3]
    y_idx=datasets[0][4]

    # X = torch.normal(0, 1, size=(2000, 30))
    # y = X[:, 16] + X[:, 17]+X[:, 26]
    # y_idx=9

    model.to("cuda")
    dag = nx.from_numpy_array(adjacency, create_using=nx.DiGraph)
    plt.figure(figsize=(8, 4))
    nx.draw(dag, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.savefig(f'dag_{y_idx}.png')

    inference_attention = InferenceAttentionMap(model_path=model_path, calculate_feature_attention=True,
                                                calculate_sample_attention=True)
    feature_attention_score, sample_attention_score = inference_attention.inference_on_single_gpu(X_full[0:1600], y_full[0:1600], X_full[1600:],"cls")
    plt.figure(figsize=(8, 4.2))
    feature_attention_score=feature_attention_score[:,-1,:]
    feature_attention_score=torch.cat((feature_attention_score[:,:y_idx],feature_attention_score[:,-1:],feature_attention_score[:,y_idx:-1]),dim=1)
    heatmap = sns.heatmap(feature_attention_score.cpu(), annot=False, fmt=".2f", cmap='YlGnBu',
                          cbar_kws={'label': 'Value'})
    output=simple_inference(model, X_full[0:1600], y_full[0:1600], X_full[1600:],"cls")
    output = output[:1600, :2].float()
    outputs = torch.nn.functional.softmax(output, dim=1)
    #
    output = outputs.float().cpu().numpy()
    prediction_ = output / output.sum(axis=1, keepdims=True)
    auc= auc_metric(y_full[1600:].cpu().numpy(), prediction_)
    plt.savefig(f"demo_showcase_{auc}.png",dpi=450,bbox_inches='tight', pad_inches=0)
    plt.close()
    #
    exit()
    # roc = tabular_metrics.auc_metric(testy,
    #                                  torch.softmax(output[:,:len(np.unique(trainy))].to(torch.float32),dim=2).squeeze().cpu().numpy())

    # classifier = TabPFNClassifier(device=data_device, ignore_pretraining_limits=True,
    #                               model_path=model_path,
    #                               feature_select_type="quantile_uni_fine",
    #                               sample_selection_type="original", calculate_sample_attention_score=False)
    # classifier.fit(trainX, trainy)
    # prediction_ = classifier.predict_proba(testX)
    #
    # roc = tabular_metrics.auc_metric(testy,
    #                                  prediction_)

    # gc.collect()
    # torch.cuda.empty_cache()
    # classifier = TabPFNClassifier(device=data_device, ignore_pretraining_limits=True,
    #                               model_path=model_path,
    #                               feature_select_type="quantile_uni_fine",
    #                               sample_selection_type="DP", calculate_feature_attention_score=True)
    # classifier.fit(trainX, trainy)
    # attention_scores = classifier.predict_proba(testX)
    # gc.collect()
    # torch.cuda.empty_cache()
    # classifier = TabPFNClassifier(device=data_device, ignore_pretraining_limits=True,
    #                               model_path=model_path,
    #                               feature_select_type="quantile_uni_fine",
    #                               sample_selection_type="AM", calculate_sample_attention_score=False,
    #                               attention_score=attention_scores)
    # classifier.fit(trainX, trainy)
    # attention_scores = classifier.predict_proba(testX)

    for layer in range(12, 13):

        sampleModel, inference_config, _ = initialize_tabpfn_model(
            model_path=model_path,
            which="classifier",
            fit_mode="fit_preprocessors",
            calculate_sample_attention_score=True, num_layer=12, inference_with_positional_embeddings=True,
            static_seed=0, use_residual_on_mlp=True, use_residual_on_items=True, use_residual_on_features=True)

        featureModel, inference_config, _ = initialize_tabpfn_model(
            model_path=model_path,
            which="classifier",
            fit_mode="fit_preprocessors",
            calculate_feature_attention_score=True, num_layer=12, inference_with_positional_embeddings=True,
            static_seed=0, use_residual_on_mlp=True, use_residual_on_items=True, use_residual_on_features=True)
        sampleModel.to("cuda")
        featureModel.to("cuda")

        data = (None, X_full, torch.tensor(trainy, dtype=torch.float32).to("cuda"))
        # data = (None, X_full, y)
        with (
            torch.autocast("cuda", enabled=True),
            torch.inference_mode()
        ):
            sampleoutput = sampleModel(
                data,
                only_return_standard_out=True,
                single_eval_pos=trainy.shape[0],
            )
            featureoutput = featureModel(
                data,
                only_return_standard_out=True,
                single_eval_pos=trainy.shape[0],
            )
        sample_attention_score = sampleoutput.cpu()
        feature_attention_score = featureoutput.cpu()  # (seq_len,num_feature,num_feature)
        y_feature_attention = feature_attention_score[sample_attention_score.shape[-1]:, -1,
                              :].squeeze()  # (train_len,num_feature)
        sample_attention_score = sample_attention_score[:, 0:10, :]  # (num_feature,10,train_len)

        for col in range(sample_attention_score.shape[1]):
            sample_attention_item = sample_attention_score[:, col, :]
            sample_attention_item = sample_attention_item.permute(1, 0)
            # attention_item=sample_attention_item
            # attention_item=sample_attention_item[:,:-1]
            attention_item = sample_attention_item[:, :] * y_feature_attention[col, :]
            attention_item = torch.mean(attention_item, dim=1, keepdim=True)
            plt.figure(figsize=(8, 6))
            heatmap = sns.heatmap(attention_item, annot=False, fmt=".2f", cmap='YlGnBu',
                                  cbar_kws={'label': 'Value'})
            os.makedirs(f"{folder_name}_item_select/layer{layer}", exist_ok=True)
            plt.savefig(f"{folder_name}_item_select/layer{layer}/{col}_item_importance.png")
            plt.close()
            distribution_type = "full shift"
            attention_item = attention_item.permute(1, 0)
            top_k_indices = np.argsort(attention_item)[:, -500:]
            other_indices = np.argsort(attention_item)[:, :-500]
            for label, sample_index in enumerate(top_k_indices):
                plot_points(trainX[sample_index], trainy[sample_index],
                            dir=f"select_sample_{distribution_type}",
                            data_type=f"top_k",
                            file_name=f"{len(np.unique(testy))}_label_{col}",
                            test_X=testX[col], test_y=output[col])
            for label, sample_index in enumerate(other_indices):
                plot_points(trainX[sample_index], trainy[sample_index],
                            dir=f"select_sample_{distribution_type}",
                            data_type="other",
                            file_name=f"{len(np.unique(testy))}_label_{col}",
                            test_X=testX[col], test_y=output[col])

