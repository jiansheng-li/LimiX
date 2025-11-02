import torch
import random
import numpy as np
from model.transformer import FeaturesTransformer

key_map = {
    "encoder.0.numeric_mlp.0.weight": "encoder_x.0.numeric_mlp.0.weight",
    "encoder.0.numeric_mlp.0.bias": "encoder_x.0.numeric_mlp.0.bias",
    "encoder.0.numeric_mlp.1.weight": "encoder_x.0.numeric_mlp.1.weight",
    "encoder.0.numeric_mlp.1.bias": "encoder_x.0.numeric_mlp.1.bias",
    "encoder.0.numeric_mlp.3.weight": "encoder_x.0.numeric_mlp.3.weight",
    "encoder.0.numeric_mlp.3.bias": "encoder_x.0.numeric_mlp.3.bias",
    "encoder.0.numeric_mlp.4.weight": "encoder_x.0.numeric_mlp.4.weight",
    "encoder.0.numeric_mlp.4.bias": "encoder_x.0.numeric_mlp.4.bias",
    "encoder.0.fusion_network.0.weight": "encoder_x.0.fusion_network.0.weight",
    "encoder.0.fusion_network.0.bias": "encoder_x.0.fusion_network.0.bias",
    "encoder.0.fusion_network.1.weight": "encoder_x.0.fusion_network.1.weight",
    "encoder.0.fusion_network.1.bias": "encoder_x.0.fusion_network.1.bias",
    "encoder.0.fusion_network.3.weight": "encoder_x.0.fusion_network.3.weight",
    "encoder.0.fusion_network.3.bias": "encoder_x.0.fusion_network.3.bias",
    "encoder.0.fusion_network.4.weight": "encoder_x.0.fusion_network.4.weight",
    "encoder.0.fusion_network.4.bias": "encoder_x.0.fusion_network.4.bias",
    "encoder.0.numeric_mlp.sigma": "encoder_x.0.numeric_mlp.sigma",
    "encoder.0.numeric_mlp.sign_embedding.weight": "encoder_x.0.numeric_mlp.sign_embedding.weight",
    "encoder.0.numeric_mlp.exp_sign_embedding.weight": "encoder_x.0.numeric_mlp.exp_sign_embedding.weight",
    "encoder.0.numeric_mlp.exp_digit_embedding.weight": "encoder_x.0.numeric_mlp.exp_digit_embedding.weight",
    "encoder.0.numeric_mlp.gate_mlp.0.weight": "encoder_x.0.numeric_mlp.gate_mlp.0.weight",
    "encoder.0.numeric_mlp.gate_mlp.0.bias": "encoder_x.0.numeric_mlp.gate_mlp.0.bias",
    "encoder.0.numeric_mlp.gate_mlp.2.weight": "encoder_x.0.numeric_mlp.gate_mlp.2.weight",
    "encoder.0.numeric_mlp.gate_mlp.2.bias": "encoder_x.0.numeric_mlp.gate_mlp.2.bias",
    "encoder.0.numeric_mlp.norm.weight": "encoder_x.0.numeric_mlp.norm.weight",
    "encoder.0.numeric_mlp.norm.bias": "encoder_x.0.numeric_mlp.norm.bias",
    "encoder.0.numeric_mlp.out_layer.weight": "encoder_x.0.numeric_mlp.out_layer.weight",
    "encoder.0.numeric_mlp.out_layer.bias": "encoder_x.0.numeric_mlp.out_layer.bias",
    "encoder.0.mask_embedding": "encoder_x.0.mask_embedding",
    "feature_positional_embedding_embeddings.weight": "feature_positional_embedding.weight",
    "feature_positional_embedding_embeddings.bias": "feature_positional_embedding.bias",
    "transformer_encoder.layers.0.self_attn_between_features._w_out": "transformer_encoder.layers.0.feature_attentions.0.out_proj_weight",
    "transformer_encoder.layers.0.self_attn_between_features._w_qkv": "transformer_encoder.layers.0.feature_attentions.0.qkv_proj_weight",
    "transformer_encoder.layers.0.self_attn_between_features_2._w_out": "transformer_encoder.layers.0.feature_attentions.1.out_proj_weight",
    "transformer_encoder.layers.0.self_attn_between_features_2._w_qkv": "transformer_encoder.layers.0.feature_attentions.1.qkv_proj_weight",
    "transformer_encoder.layers.0.self_attn_between_items._w_out": "transformer_encoder.layers.0.sequence_attentions.0.out_proj_weight",
    "transformer_encoder.layers.0.self_attn_between_items._w_qkv": "transformer_encoder.layers.0.sequence_attentions.0.qkv_proj_weight",
    "transformer_encoder.layers.0.mlp.linear1.weight": "transformer_encoder.layers.0.mlp.0.mlp.0.weight",
    "transformer_encoder.layers.0.mlp.linear2.weight": "transformer_encoder.layers.0.mlp.0.mlp.2.weight",
    "transformer_encoder.layers.0.mlp2.linear1.weight": "transformer_encoder.layers.0.mlp.1.mlp.0.weight",
    "transformer_encoder.layers.0.mlp2.linear2.weight": "transformer_encoder.layers.0.mlp.1.mlp.2.weight",
    "transformer_encoder.layers.0.mlp3.linear1.weight": "transformer_encoder.layers.0.mlp.2.mlp.0.weight",
    "transformer_encoder.layers.0.mlp3.linear2.weight": "transformer_encoder.layers.0.mlp.2.mlp.2.weight",
    "transformer_encoder.layers.1.self_attn_between_features._w_out": "transformer_encoder.layers.1.feature_attentions.0.out_proj_weight",
    "transformer_encoder.layers.1.self_attn_between_features._w_qkv": "transformer_encoder.layers.1.feature_attentions.0.qkv_proj_weight",
    "transformer_encoder.layers.1.self_attn_between_features_2._w_out": "transformer_encoder.layers.1.feature_attentions.1.out_proj_weight",
    "transformer_encoder.layers.1.self_attn_between_features_2._w_qkv": "transformer_encoder.layers.1.feature_attentions.1.qkv_proj_weight",
    "transformer_encoder.layers.1.self_attn_between_items._w_out": "transformer_encoder.layers.1.sequence_attentions.0.out_proj_weight",
    "transformer_encoder.layers.1.self_attn_between_items._w_qkv": "transformer_encoder.layers.1.sequence_attentions.0.qkv_proj_weight",
    "transformer_encoder.layers.1.mlp.linear1.weight": "transformer_encoder.layers.1.mlp.0.mlp.0.weight",
    "transformer_encoder.layers.1.mlp.linear2.weight": "transformer_encoder.layers.1.mlp.0.mlp.2.weight",
    "transformer_encoder.layers.1.mlp2.linear1.weight": "transformer_encoder.layers.1.mlp.1.mlp.0.weight",
    "transformer_encoder.layers.1.mlp2.linear2.weight": "transformer_encoder.layers.1.mlp.1.mlp.2.weight",
    "transformer_encoder.layers.1.mlp3.linear1.weight": "transformer_encoder.layers.1.mlp.2.mlp.0.weight",
    "transformer_encoder.layers.1.mlp3.linear2.weight": "transformer_encoder.layers.1.mlp.2.mlp.2.weight",
    "transformer_encoder.layers.2.self_attn_between_features._w_out": "transformer_encoder.layers.2.feature_attentions.0.out_proj_weight",
    "transformer_encoder.layers.2.self_attn_between_features._w_qkv": "transformer_encoder.layers.2.feature_attentions.0.qkv_proj_weight",
    "transformer_encoder.layers.2.self_attn_between_features_2._w_out": "transformer_encoder.layers.2.feature_attentions.1.out_proj_weight",
    "transformer_encoder.layers.2.self_attn_between_features_2._w_qkv": "transformer_encoder.layers.2.feature_attentions.1.qkv_proj_weight",
    "transformer_encoder.layers.2.self_attn_between_items._w_out": "transformer_encoder.layers.2.sequence_attentions.0.out_proj_weight",
    "transformer_encoder.layers.2.self_attn_between_items._w_qkv": "transformer_encoder.layers.2.sequence_attentions.0.qkv_proj_weight",
    "transformer_encoder.layers.2.mlp.linear1.weight": "transformer_encoder.layers.2.mlp.0.mlp.0.weight",
    "transformer_encoder.layers.2.mlp.linear2.weight": "transformer_encoder.layers.2.mlp.0.mlp.2.weight",
    "transformer_encoder.layers.2.mlp2.linear1.weight": "transformer_encoder.layers.2.mlp.1.mlp.0.weight",
    "transformer_encoder.layers.2.mlp2.linear2.weight": "transformer_encoder.layers.2.mlp.1.mlp.2.weight",
    "transformer_encoder.layers.2.mlp3.linear1.weight": "transformer_encoder.layers.2.mlp.2.mlp.0.weight",
    "transformer_encoder.layers.2.mlp3.linear2.weight": "transformer_encoder.layers.2.mlp.2.mlp.2.weight",
    "transformer_encoder.layers.3.self_attn_between_features._w_out": "transformer_encoder.layers.3.feature_attentions.0.out_proj_weight",
    "transformer_encoder.layers.3.self_attn_between_features._w_qkv": "transformer_encoder.layers.3.feature_attentions.0.qkv_proj_weight",
    "transformer_encoder.layers.3.self_attn_between_features_2._w_out": "transformer_encoder.layers.3.feature_attentions.1.out_proj_weight",
    "transformer_encoder.layers.3.self_attn_between_features_2._w_qkv": "transformer_encoder.layers.3.feature_attentions.1.qkv_proj_weight",
    "transformer_encoder.layers.3.self_attn_between_items._w_out": "transformer_encoder.layers.3.sequence_attentions.0.out_proj_weight",
    "transformer_encoder.layers.3.self_attn_between_items._w_qkv": "transformer_encoder.layers.3.sequence_attentions.0.qkv_proj_weight",
    "transformer_encoder.layers.3.mlp.linear1.weight": "transformer_encoder.layers.3.mlp.0.mlp.0.weight",
    "transformer_encoder.layers.3.mlp.linear2.weight": "transformer_encoder.layers.3.mlp.0.mlp.2.weight",
    "transformer_encoder.layers.3.mlp2.linear1.weight": "transformer_encoder.layers.3.mlp.1.mlp.0.weight",
    "transformer_encoder.layers.3.mlp2.linear2.weight": "transformer_encoder.layers.3.mlp.1.mlp.2.weight",
    "transformer_encoder.layers.3.mlp3.linear1.weight": "transformer_encoder.layers.3.mlp.2.mlp.0.weight",
    "transformer_encoder.layers.3.mlp3.linear2.weight": "transformer_encoder.layers.3.mlp.2.mlp.2.weight",
    "transformer_encoder.layers.4.self_attn_between_features._w_out": "transformer_encoder.layers.4.feature_attentions.0.out_proj_weight",
    "transformer_encoder.layers.4.self_attn_between_features._w_qkv": "transformer_encoder.layers.4.feature_attentions.0.qkv_proj_weight",
    "transformer_encoder.layers.4.self_attn_between_features_2._w_out": "transformer_encoder.layers.4.feature_attentions.1.out_proj_weight",
    "transformer_encoder.layers.4.self_attn_between_features_2._w_qkv": "transformer_encoder.layers.4.feature_attentions.1.qkv_proj_weight",
    "transformer_encoder.layers.4.self_attn_between_items._w_out": "transformer_encoder.layers.4.sequence_attentions.0.out_proj_weight",
    "transformer_encoder.layers.4.self_attn_between_items._w_qkv": "transformer_encoder.layers.4.sequence_attentions.0.qkv_proj_weight",
    "transformer_encoder.layers.4.mlp.linear1.weight": "transformer_encoder.layers.4.mlp.0.mlp.0.weight",
    "transformer_encoder.layers.4.mlp.linear2.weight": "transformer_encoder.layers.4.mlp.0.mlp.2.weight",
    "transformer_encoder.layers.4.mlp2.linear1.weight": "transformer_encoder.layers.4.mlp.1.mlp.0.weight",
    "transformer_encoder.layers.4.mlp2.linear2.weight": "transformer_encoder.layers.4.mlp.1.mlp.2.weight",
    "transformer_encoder.layers.4.mlp3.linear1.weight": "transformer_encoder.layers.4.mlp.2.mlp.0.weight",
    "transformer_encoder.layers.4.mlp3.linear2.weight": "transformer_encoder.layers.4.mlp.2.mlp.2.weight",
    "transformer_encoder.layers.5.self_attn_between_features._w_out": "transformer_encoder.layers.5.feature_attentions.0.out_proj_weight",
    "transformer_encoder.layers.5.self_attn_between_features._w_qkv": "transformer_encoder.layers.5.feature_attentions.0.qkv_proj_weight",
    "transformer_encoder.layers.5.self_attn_between_features_2._w_out": "transformer_encoder.layers.5.feature_attentions.1.out_proj_weight",
    "transformer_encoder.layers.5.self_attn_between_features_2._w_qkv": "transformer_encoder.layers.5.feature_attentions.1.qkv_proj_weight",
    "transformer_encoder.layers.5.self_attn_between_items._w_out": "transformer_encoder.layers.5.sequence_attentions.0.out_proj_weight",
    "transformer_encoder.layers.5.self_attn_between_items._w_qkv": "transformer_encoder.layers.5.sequence_attentions.0.qkv_proj_weight",
    "transformer_encoder.layers.5.mlp.linear1.weight": "transformer_encoder.layers.5.mlp.0.mlp.0.weight",
    "transformer_encoder.layers.5.mlp.linear2.weight": "transformer_encoder.layers.5.mlp.0.mlp.2.weight",
    "transformer_encoder.layers.5.mlp2.linear1.weight": "transformer_encoder.layers.5.mlp.1.mlp.0.weight",
    "transformer_encoder.layers.5.mlp2.linear2.weight": "transformer_encoder.layers.5.mlp.1.mlp.2.weight",
    "transformer_encoder.layers.5.mlp3.linear1.weight": "transformer_encoder.layers.5.mlp.2.mlp.0.weight",
    "transformer_encoder.layers.5.mlp3.linear2.weight": "transformer_encoder.layers.5.mlp.2.mlp.2.weight",
    "transformer_encoder.layers.6.self_attn_between_features._w_out": "transformer_encoder.layers.6.feature_attentions.0.out_proj_weight",
    "transformer_encoder.layers.6.self_attn_between_features._w_qkv": "transformer_encoder.layers.6.feature_attentions.0.qkv_proj_weight",
    "transformer_encoder.layers.6.self_attn_between_features_2._w_out": "transformer_encoder.layers.6.feature_attentions.1.out_proj_weight",
    "transformer_encoder.layers.6.self_attn_between_features_2._w_qkv": "transformer_encoder.layers.6.feature_attentions.1.qkv_proj_weight",
    "transformer_encoder.layers.6.self_attn_between_items._w_out": "transformer_encoder.layers.6.sequence_attentions.0.out_proj_weight",
    "transformer_encoder.layers.6.self_attn_between_items._w_qkv": "transformer_encoder.layers.6.sequence_attentions.0.qkv_proj_weight",
    "transformer_encoder.layers.6.mlp.linear1.weight": "transformer_encoder.layers.6.mlp.0.mlp.0.weight",
    "transformer_encoder.layers.6.mlp.linear2.weight": "transformer_encoder.layers.6.mlp.0.mlp.2.weight",
    "transformer_encoder.layers.6.mlp2.linear1.weight": "transformer_encoder.layers.6.mlp.1.mlp.0.weight",
    "transformer_encoder.layers.6.mlp2.linear2.weight": "transformer_encoder.layers.6.mlp.1.mlp.2.weight",
    "transformer_encoder.layers.6.mlp3.linear1.weight": "transformer_encoder.layers.6.mlp.2.mlp.0.weight",
    "transformer_encoder.layers.6.mlp3.linear2.weight": "transformer_encoder.layers.6.mlp.2.mlp.2.weight",
    "transformer_encoder.layers.7.self_attn_between_features._w_out": "transformer_encoder.layers.7.feature_attentions.0.out_proj_weight",
    "transformer_encoder.layers.7.self_attn_between_features._w_qkv": "transformer_encoder.layers.7.feature_attentions.0.qkv_proj_weight",
    "transformer_encoder.layers.7.self_attn_between_features_2._w_out": "transformer_encoder.layers.7.feature_attentions.1.out_proj_weight",
    "transformer_encoder.layers.7.self_attn_between_features_2._w_qkv": "transformer_encoder.layers.7.feature_attentions.1.qkv_proj_weight",
    "transformer_encoder.layers.7.self_attn_between_items._w_out": "transformer_encoder.layers.7.sequence_attentions.0.out_proj_weight",
    "transformer_encoder.layers.7.self_attn_between_items._w_qkv": "transformer_encoder.layers.7.sequence_attentions.0.qkv_proj_weight",
    "transformer_encoder.layers.7.mlp.linear1.weight": "transformer_encoder.layers.7.mlp.0.mlp.0.weight",
    "transformer_encoder.layers.7.mlp.linear2.weight": "transformer_encoder.layers.7.mlp.0.mlp.2.weight",
    "transformer_encoder.layers.7.mlp2.linear1.weight": "transformer_encoder.layers.7.mlp.1.mlp.0.weight",
    "transformer_encoder.layers.7.mlp2.linear2.weight": "transformer_encoder.layers.7.mlp.1.mlp.2.weight",
    "transformer_encoder.layers.7.mlp3.linear1.weight": "transformer_encoder.layers.7.mlp.2.mlp.0.weight",
    "transformer_encoder.layers.7.mlp3.linear2.weight": "transformer_encoder.layers.7.mlp.2.mlp.2.weight",
    "transformer_encoder.layers.8.self_attn_between_features._w_out": "transformer_encoder.layers.8.feature_attentions.0.out_proj_weight",
    "transformer_encoder.layers.8.self_attn_between_features._w_qkv": "transformer_encoder.layers.8.feature_attentions.0.qkv_proj_weight",
    "transformer_encoder.layers.8.self_attn_between_features_2._w_out": "transformer_encoder.layers.8.feature_attentions.1.out_proj_weight",
    "transformer_encoder.layers.8.self_attn_between_features_2._w_qkv": "transformer_encoder.layers.8.feature_attentions.1.qkv_proj_weight",
    "transformer_encoder.layers.8.self_attn_between_items._w_out": "transformer_encoder.layers.8.sequence_attentions.0.out_proj_weight",
    "transformer_encoder.layers.8.self_attn_between_items._w_qkv": "transformer_encoder.layers.8.sequence_attentions.0.qkv_proj_weight",
    "transformer_encoder.layers.8.mlp.linear1.weight": "transformer_encoder.layers.8.mlp.0.mlp.0.weight",
    "transformer_encoder.layers.8.mlp.linear2.weight": "transformer_encoder.layers.8.mlp.0.mlp.2.weight",
    "transformer_encoder.layers.8.mlp2.linear1.weight": "transformer_encoder.layers.8.mlp.1.mlp.0.weight",
    "transformer_encoder.layers.8.mlp2.linear2.weight": "transformer_encoder.layers.8.mlp.1.mlp.2.weight",
    "transformer_encoder.layers.8.mlp3.linear1.weight": "transformer_encoder.layers.8.mlp.2.mlp.0.weight",
    "transformer_encoder.layers.8.mlp3.linear2.weight": "transformer_encoder.layers.8.mlp.2.mlp.2.weight",
    "transformer_encoder.layers.9.self_attn_between_features._w_out": "transformer_encoder.layers.9.feature_attentions.0.out_proj_weight",
    "transformer_encoder.layers.9.self_attn_between_features._w_qkv": "transformer_encoder.layers.9.feature_attentions.0.qkv_proj_weight",
    "transformer_encoder.layers.9.self_attn_between_features_2._w_out": "transformer_encoder.layers.9.feature_attentions.1.out_proj_weight",
    "transformer_encoder.layers.9.self_attn_between_features_2._w_qkv": "transformer_encoder.layers.9.feature_attentions.1.qkv_proj_weight",
    "transformer_encoder.layers.9.self_attn_between_items._w_out": "transformer_encoder.layers.9.sequence_attentions.0.out_proj_weight",
    "transformer_encoder.layers.9.self_attn_between_items._w_qkv": "transformer_encoder.layers.9.sequence_attentions.0.qkv_proj_weight",
    "transformer_encoder.layers.9.mlp.linear1.weight": "transformer_encoder.layers.9.mlp.0.mlp.0.weight",
    "transformer_encoder.layers.9.mlp.linear2.weight": "transformer_encoder.layers.9.mlp.0.mlp.2.weight",
    "transformer_encoder.layers.9.mlp2.linear1.weight": "transformer_encoder.layers.9.mlp.1.mlp.0.weight",
    "transformer_encoder.layers.9.mlp2.linear2.weight": "transformer_encoder.layers.9.mlp.1.mlp.2.weight",
    "transformer_encoder.layers.9.mlp3.linear1.weight": "transformer_encoder.layers.9.mlp.2.mlp.0.weight",
    "transformer_encoder.layers.9.mlp3.linear2.weight": "transformer_encoder.layers.9.mlp.2.mlp.2.weight",
    "transformer_encoder.layers.10.self_attn_between_features._w_out": "transformer_encoder.layers.10.feature_attentions.0.out_proj_weight",
    "transformer_encoder.layers.10.self_attn_between_features._w_qkv": "transformer_encoder.layers.10.feature_attentions.0.qkv_proj_weight",
    "transformer_encoder.layers.10.self_attn_between_features_2._w_out": "transformer_encoder.layers.10.feature_attentions.1.out_proj_weight",
    "transformer_encoder.layers.10.self_attn_between_features_2._w_qkv": "transformer_encoder.layers.10.feature_attentions.1.qkv_proj_weight",
    "transformer_encoder.layers.10.self_attn_between_items._w_out": "transformer_encoder.layers.10.sequence_attentions.0.out_proj_weight",
    "transformer_encoder.layers.10.self_attn_between_items._w_qkv": "transformer_encoder.layers.10.sequence_attentions.0.qkv_proj_weight",
    "transformer_encoder.layers.10.mlp.linear1.weight": "transformer_encoder.layers.10.mlp.0.mlp.0.weight",
    "transformer_encoder.layers.10.mlp.linear2.weight": "transformer_encoder.layers.10.mlp.0.mlp.2.weight",
    "transformer_encoder.layers.10.mlp2.linear1.weight": "transformer_encoder.layers.10.mlp.1.mlp.0.weight",
    "transformer_encoder.layers.10.mlp2.linear2.weight": "transformer_encoder.layers.10.mlp.1.mlp.2.weight",
    "transformer_encoder.layers.10.mlp3.linear1.weight": "transformer_encoder.layers.10.mlp.2.mlp.0.weight",
    "transformer_encoder.layers.10.mlp3.linear2.weight": "transformer_encoder.layers.10.mlp.2.mlp.2.weight",
    "transformer_encoder.layers.11.self_attn_between_features._w_out": "transformer_encoder.layers.11.feature_attentions.0.out_proj_weight",
    "transformer_encoder.layers.11.self_attn_between_features._w_qkv": "transformer_encoder.layers.11.feature_attentions.0.qkv_proj_weight",
    "transformer_encoder.layers.11.self_attn_between_features_2._w_out": "transformer_encoder.layers.11.feature_attentions.1.out_proj_weight",
    "transformer_encoder.layers.11.self_attn_between_features_2._w_qkv": "transformer_encoder.layers.11.feature_attentions.1.qkv_proj_weight",
    "transformer_encoder.layers.11.self_attn_between_items._w_out": "transformer_encoder.layers.11.sequence_attentions.0.out_proj_weight",
    "transformer_encoder.layers.11.self_attn_between_items._w_qkv": "transformer_encoder.layers.11.sequence_attentions.0.qkv_proj_weight",
    "transformer_encoder.layers.11.mlp.linear1.weight": "transformer_encoder.layers.11.mlp.0.mlp.0.weight",
    "transformer_encoder.layers.11.mlp.linear2.weight": "transformer_encoder.layers.11.mlp.0.mlp.2.weight",
    "transformer_encoder.layers.11.mlp2.linear1.weight": "transformer_encoder.layers.11.mlp.1.mlp.0.weight",
    "transformer_encoder.layers.11.mlp2.linear2.weight": "transformer_encoder.layers.11.mlp.1.mlp.2.weight",
    "transformer_encoder.layers.11.mlp3.linear1.weight": "transformer_encoder.layers.11.mlp.2.mlp.0.weight",
    "transformer_encoder.layers.11.mlp3.linear2.weight": "transformer_encoder.layers.11.mlp.2.mlp.2.weight",

    # 2M模型新增
    "transformer_encoder.layers.0.self_attn_between_items_1._w_out": "transformer_encoder.layers.0.sequence_attentions.0.out_proj_weight",
    "transformer_encoder.layers.0.self_attn_between_items_1._w_qkv": "transformer_encoder.layers.0.sequence_attentions.0.qkv_proj_weight",
    "transformer_encoder.layers.0.self_attn_between_items_2._w_out": "transformer_encoder.layers.0.sequence_attentions.1.out_proj_weight",
    "transformer_encoder.layers.0.self_attn_between_items_2._w_qkv": "transformer_encoder.layers.0.sequence_attentions.1.qkv_proj_weight",
    "transformer_encoder.layers.1.self_attn_between_items_1._w_out": "transformer_encoder.layers.1.sequence_attentions.0.out_proj_weight",
    "transformer_encoder.layers.1.self_attn_between_items_1._w_qkv": "transformer_encoder.layers.1.sequence_attentions.0.qkv_proj_weight",
    "transformer_encoder.layers.1.self_attn_between_items_2._w_out": "transformer_encoder.layers.1.sequence_attentions.1.out_proj_weight",
    "transformer_encoder.layers.1.self_attn_between_items_2._w_qkv": "transformer_encoder.layers.1.sequence_attentions.1.qkv_proj_weight",
    "transformer_encoder.layers.2.self_attn_between_items_1._w_out": "transformer_encoder.layers.2.sequence_attentions.0.out_proj_weight",
    "transformer_encoder.layers.2.self_attn_between_items_1._w_qkv": "transformer_encoder.layers.2.sequence_attentions.0.qkv_proj_weight",
    "transformer_encoder.layers.2.self_attn_between_items_2._w_out": "transformer_encoder.layers.2.sequence_attentions.1.out_proj_weight",
    "transformer_encoder.layers.2.self_attn_between_items_2._w_qkv": "transformer_encoder.layers.2.sequence_attentions.1.qkv_proj_weight",
    "transformer_encoder.layers.3.self_attn_between_items_1._w_out": "transformer_encoder.layers.3.sequence_attentions.0.out_proj_weight",
    "transformer_encoder.layers.3.self_attn_between_items_1._w_qkv": "transformer_encoder.layers.3.sequence_attentions.0.qkv_proj_weight",
    "transformer_encoder.layers.3.self_attn_between_items_2._w_out": "transformer_encoder.layers.3.sequence_attentions.1.out_proj_weight",
    "transformer_encoder.layers.3.self_attn_between_items_2._w_qkv": "transformer_encoder.layers.3.sequence_attentions.1.qkv_proj_weight",
    "transformer_encoder.layers.4.self_attn_between_items_1._w_out": "transformer_encoder.layers.4.sequence_attentions.0.out_proj_weight",
    "transformer_encoder.layers.4.self_attn_between_items_1._w_qkv": "transformer_encoder.layers.4.sequence_attentions.0.qkv_proj_weight",
    "transformer_encoder.layers.4.self_attn_between_items_2._w_out": "transformer_encoder.layers.4.sequence_attentions.1.out_proj_weight",
    "transformer_encoder.layers.4.self_attn_between_items_2._w_qkv": "transformer_encoder.layers.4.sequence_attentions.1.qkv_proj_weight",
    "transformer_encoder.layers.5.self_attn_between_items_1._w_out": "transformer_encoder.layers.5.sequence_attentions.0.out_proj_weight",
    "transformer_encoder.layers.5.self_attn_between_items_1._w_qkv": "transformer_encoder.layers.5.sequence_attentions.0.qkv_proj_weight",
    "transformer_encoder.layers.5.self_attn_between_items_2._w_out": "transformer_encoder.layers.5.sequence_attentions.1.out_proj_weight",
    "transformer_encoder.layers.5.self_attn_between_items_2._w_qkv": "transformer_encoder.layers.5.sequence_attentions.1.qkv_proj_weight",
    "transformer_encoder.layers.6.self_attn_between_items_1._w_out": "transformer_encoder.layers.6.sequence_attentions.0.out_proj_weight",
    "transformer_encoder.layers.6.self_attn_between_items_1._w_qkv": "transformer_encoder.layers.6.sequence_attentions.0.qkv_proj_weight",
    "transformer_encoder.layers.6.self_attn_between_items_2._w_out": "transformer_encoder.layers.6.sequence_attentions.1.out_proj_weight",
    "transformer_encoder.layers.6.self_attn_between_items_2._w_qkv": "transformer_encoder.layers.6.sequence_attentions.1.qkv_proj_weight",
    "transformer_encoder.layers.7.self_attn_between_items_1._w_out": "transformer_encoder.layers.7.sequence_attentions.0.out_proj_weight",
    "transformer_encoder.layers.7.self_attn_between_items_1._w_qkv": "transformer_encoder.layers.7.sequence_attentions.0.qkv_proj_weight",
    "transformer_encoder.layers.7.self_attn_between_items_2._w_out": "transformer_encoder.layers.7.sequence_attentions.1.out_proj_weight",
    "transformer_encoder.layers.7.self_attn_between_items_2._w_qkv": "transformer_encoder.layers.7.sequence_attentions.1.qkv_proj_weight",
    "transformer_encoder.layers.8.self_attn_between_items_1._w_out": "transformer_encoder.layers.8.sequence_attentions.0.out_proj_weight",
    "transformer_encoder.layers.8.self_attn_between_items_1._w_qkv": "transformer_encoder.layers.8.sequence_attentions.0.qkv_proj_weight",
    "transformer_encoder.layers.8.self_attn_between_items_2._w_out": "transformer_encoder.layers.8.sequence_attentions.1.out_proj_weight",
    "transformer_encoder.layers.8.self_attn_between_items_2._w_qkv": "transformer_encoder.layers.8.sequence_attentions.1.qkv_proj_weight",
    "transformer_encoder.layers.9.self_attn_between_items_1._w_out": "transformer_encoder.layers.9.sequence_attentions.0.out_proj_weight",
    "transformer_encoder.layers.9.self_attn_between_items_1._w_qkv": "transformer_encoder.layers.9.sequence_attentions.0.qkv_proj_weight",
    "transformer_encoder.layers.9.self_attn_between_items_2._w_out": "transformer_encoder.layers.9.sequence_attentions.1.out_proj_weight",
    "transformer_encoder.layers.9.self_attn_between_items_2._w_qkv": "transformer_encoder.layers.9.sequence_attentions.1.qkv_proj_weight",
    "transformer_encoder.layers.10.self_attn_between_items_1._w_out": "transformer_encoder.layers.10.sequence_attentions.0.out_proj_weight",
    "transformer_encoder.layers.10.self_attn_between_items_1._w_qkv": "transformer_encoder.layers.10.sequence_attentions.0.qkv_proj_weight",
    "transformer_encoder.layers.10.self_attn_between_items_2._w_out": "transformer_encoder.layers.10.sequence_attentions.1.out_proj_weight",
    "transformer_encoder.layers.10.self_attn_between_items_2._w_qkv": "transformer_encoder.layers.10.sequence_attentions.1.qkv_proj_weight",
    "transformer_encoder.layers.11.self_attn_between_items_1._w_out": "transformer_encoder.layers.11.sequence_attentions.0.out_proj_weight",
    "transformer_encoder.layers.11.self_attn_between_items_1._w_qkv": "transformer_encoder.layers.11.sequence_attentions.0.qkv_proj_weight",
    "transformer_encoder.layers.11.self_attn_between_items_2._w_out": "transformer_encoder.layers.11.sequence_attentions.1.out_proj_weight",
    "transformer_encoder.layers.11.self_attn_between_items_2._w_qkv": "transformer_encoder.layers.11.sequence_attentions.1.qkv_proj_weight",

}


def state_dict_map(state: dict):
    new_state = {}
    for k, v in state.items():
        if 'criterion' in k:
            continue
        k = k.replace('module.', '')
        if k in key_map:
            new_state[key_map[k]] = v
        else:
            new_state[k] = v
    return new_state


def get_config_value(key: str, state_dict: dict | None = None, model_config: dict | None = None,
                     train_config: dict | None = None, default: any = None):
    '''获取模型配置中的值，
    若model_config、train_config存在，表示为新模型，从配置中直接获取
    若model_config、train_config为None，则尝试从权重中的config_sample中获取(老模型结构)，
    若config_sample不存在（新模型），则直接从权重中的config中获取
    '''
    if model_config is not None and train_config is not None:
        if key in model_config:
            return model_config[key]
        elif key in train_config:
            return train_config[key]
        elif default is not None:
            return default
        else:
            raise KeyError(f"Key {key} not found in model_config and train_config")
    elif state_dict is not None:
        if 'config_sample' in state_dict:
            # 老版本的模型
            # 新老key转换
            if key == "mask_feature_embedding_type":
                key = 'feature_embedding_type'
            if key == 'feature_positional_embedding_type':
                key = 'feature_positional_embedding'
            if key == 'yemb_freeze_type':
                key = 'yemb_type'
            if key == 'dropout':
                return 0
            if key == 'pre_norm':
                key = 'use_pre_norm'
            if key == 'recompute_attn':
                default = False
            if key == 'item_attn_serial':
                key = 'serial_attn'
            if key == 'numeric_embed_type':
                key = 'scino_embed_type'
            if key == 'mask_prediction':
                return True
            if key == 'seq_attn_serial':
                key = 'serial_attn'
            if key == 'seq_attn_isolated':
                key = 'item_attn_isolated'

            value = state_dict['config_sample'][key] if default is None else state_dict['config_sample'].get(key,
                                                                                                             default)

            if key == 'layer_arch':  # 处理layer_arch特殊情况
                if value == 'fmfmim':
                    value = 'fmfmsm'
                if value == 'imf':
                    value = 'smf'
                if value == 'fim':
                    value = 'fsm'

        elif 'config' in state_dict:
            # 新版本的模型
            if key == 'dropout':
                return 0
            if key == 'recompute_attn':
                default = False
            try:
                value = state_dict['config'][key] if default is None else state_dict['config'].get(key, default)
            except:
                if key == 'emsize':
                    key = 'embed_dim'
                try:
                    value = state_dict['config'][key] if default is None else state_dict['config'].get(key, default)
                except:
                    raise ValueError(f"Key {key} not found in config")
        else:
            raise ValueError(f"Key {key} not found in config and config_sample")
    else:
        raise ValueError(
            f"model_config and train_config are None, and state_dict is None, cannot get config value for key {key}")
    return value


def get_model_config(state_dict: dict | None = None, model_config: dict | None = None,
                     train_config: dict | None = None):
    config = {}

    config['preprocess_config_x'] = {
        "num_features": get_config_value(key='features_per_group', state_dict=state_dict, model_config=model_config,
                                         train_config=train_config),
        "nan_handling_enabled": get_config_value(key='nan_handling_enabled', default=True, state_dict=state_dict,
                                                 model_config=model_config, train_config=train_config),
        "normalize_on_train_only": get_config_value(key='normalize_on_train_only', default=True, state_dict=state_dict,
                                                    model_config=model_config, train_config=train_config),
        "normalize_x": get_config_value(key='normalize_x', default=True, state_dict=state_dict,
                                        model_config=model_config, train_config=train_config),
        "remove_outliers": get_config_value(key='remove_outliers', default=False, state_dict=state_dict,
                                            model_config=model_config, train_config=train_config),
        "normalize_by_used_features": get_config_value(key='normalize_by_used_features', default=True,
                                                       state_dict=state_dict, model_config=model_config,
                                                       train_config=train_config),
    }

    numeric_embed_type = get_config_value(key='numeric_embed_type', default='linear', state_dict=state_dict,
                                          model_config=model_config, train_config=train_config)
    if numeric_embed_type == 'RBF':
        RBF_config = {
            "token_embed_dim": get_config_value(key='RBF_token_embed_dim', state_dict=state_dict,
                                                model_config=model_config, train_config=train_config),
            "n_kernels": get_config_value(key='RBF_n_kernels', state_dict=state_dict, model_config=model_config,
                                          train_config=train_config),
            "sigma": get_config_value(key='RBF_sigma', state_dict=state_dict, model_config=model_config,
                                      train_config=train_config),
            "use_learn_sigma": get_config_value(key='RBF_use_learn_sigma', state_dict=state_dict,
                                                model_config=model_config, train_config=train_config),
            "init_embedding": get_config_value(key='RBF_init_embedding', state_dict=state_dict,
                                               model_config=model_config, train_config=train_config),
            "use_learn_embeddings": get_config_value(key='RBF_use_learn_embeddings', state_dict=state_dict,
                                                     model_config=model_config, train_config=train_config),
            "use_random_kernels": get_config_value(key='RBF_use_random_kernels', state_dict=state_dict,
                                                   model_config=model_config, train_config=train_config),
            "use_original_features": get_config_value(key='RBF_use_original_features', state_dict=state_dict,
                                                      model_config=model_config, train_config=train_config),
        }
    else:
        RBF_config = None
    emsize = get_config_value(key='emsize', state_dict=state_dict, model_config=model_config, train_config=train_config)
    config['encoder_config_x'] = {
        "num_features": get_config_value(key='features_per_group', state_dict=state_dict, model_config=model_config,
                                         train_config=train_config),
        "embedding_size": emsize,
        "mask_embedding_size": get_config_value(key='mask_embedding_size', default=emsize, state_dict=state_dict,
                                                model_config=model_config, train_config=train_config),
        "encoder_use_bias": get_config_value(key='encoder_use_bias', default=True, state_dict=state_dict,
                                             model_config=model_config, train_config=train_config),
        "feature_embedding_type": get_config_value(key='mask_feature_embedding_type', default="mask_embedding",
                                                   state_dict=state_dict, model_config=model_config,
                                                   train_config=train_config),
        "numeric_embed_type": numeric_embed_type,
        "RBF_config": RBF_config,
    }
    config['encoder_config_y'] = {
        "num_inputs": 1,
        "embedding_size": get_config_value(key='emsize', state_dict=state_dict, model_config=model_config,
                                           train_config=train_config),
        "nan_handling_y_encoder": get_config_value(key='nan_handling_y_encoder', default=True, state_dict=state_dict,
                                                   model_config=model_config, train_config=train_config),
        "max_num_classes": get_config_value(key='max_num_classes', default=10, state_dict=state_dict,
                                            model_config=model_config, train_config=train_config),
        "yemb_freeze_type": get_config_value(key='yemb_freeze_type', default='none', state_dict=state_dict,
                                             model_config=model_config, train_config=train_config),
    }
    config['decoder_config'] = {
        "num_classes": get_config_value(key='max_num_classes', default=10, state_dict=state_dict,
                                        model_config=model_config, train_config=train_config),
    }
    config['feature_positional_embedding_type'] = get_config_value(key='feature_positional_embedding_type',
                                                                   default='subortho', state_dict=state_dict,
                                                                   model_config=model_config, train_config=train_config)
    config['classify_reg_mixed'] = get_config_value(key='classify_reg_mixed', default=True, state_dict=state_dict,
                                                    model_config=model_config, train_config=train_config)
    config['nlayers'] = get_config_value(key='nlayers', state_dict=state_dict, model_config=model_config,
                                         train_config=train_config)
    config['nhead'] = get_config_value(key='nhead', state_dict=state_dict, model_config=model_config,
                                       train_config=train_config)
    config['embed_dim'] = get_config_value(key='emsize', state_dict=state_dict, model_config=model_config,
                                           train_config=train_config)
    config['hid_dim'] = get_config_value(key='emsize', state_dict=state_dict, model_config=model_config,
                                         train_config=train_config) * get_config_value(key='nhid_factor', default=4,
                                                                                       state_dict=state_dict,
                                                                                       model_config=model_config,
                                                                                       train_config=train_config)
    config['mask_feature_embedding_type'] = get_config_value(key='mask_feature_embedding_type',
                                                             default='mask_embedding', state_dict=state_dict,
                                                             model_config=model_config, train_config=train_config)
    config['enable_mask_feature_pred'] = get_config_value(key='enable_mask_feature_pred', default=True,
                                                          state_dict=state_dict, model_config=model_config,
                                                          train_config=train_config)
    config['enable_mask_indicator_pred'] = get_config_value(key='enable_mask_indicator_pred', default=False,
                                                            state_dict=state_dict, model_config=model_config,
                                                            train_config=train_config)
    config['mask_prediction'] = get_config_value(key='mask_prediction', default=False, state_dict=state_dict,
                                                 model_config=model_config, train_config=train_config)
    config['features_per_group'] = get_config_value(key='features_per_group', state_dict=state_dict,
                                                    model_config=model_config, train_config=train_config)
    config['dropout'] = get_config_value(key='dropout', state_dict=state_dict, model_config=model_config,
                                         train_config=train_config)
    config['pre_norm'] = get_config_value(key='pre_norm', default=True, state_dict=state_dict,
                                          model_config=model_config, train_config=train_config)
    config['activation'] = get_config_value(key='activation', default='gelu', state_dict=state_dict,
                                            model_config=model_config, train_config=train_config)
    config['recompute_attn'] = get_config_value(key='recompute_attn', state_dict=state_dict, model_config=model_config,
                                                train_config=train_config)
    config['qkv_w_init_method'] = get_config_value(key='qkv_w_init_method', default='xavier_uniform',
                                                   state_dict=state_dict, model_config=model_config,
                                                   train_config=train_config)
    config['mlp_init_std'] = get_config_value(key='init_std', default=0, state_dict=state_dict,
                                              model_config=model_config, train_config=train_config)
    config['mlp_use_residual'] = get_config_value(key='mlp_use_residual', default=True, state_dict=state_dict,
                                                  model_config=model_config, train_config=train_config)
    config['layer_arch'] = get_config_value(key='layer_arch', default='fmfmsm', state_dict=state_dict,
                                            model_config=model_config, train_config=train_config)

    config['calculate_feature_attention'] = get_config_value(key='calculate_feature_attention', default=False,
                                                             state_dict=state_dict, model_config=model_config,
                                                             train_config=train_config)
    config['calculate_sample_attention'] = get_config_value(key='calculate_sample_attention', default=False,
                                                            state_dict=state_dict, model_config=model_config,
                                                            train_config=train_config)

    config['self_share_all_kv_heads'] = get_config_value(key='self_share_all_kv_heads', default=False,
                                                         state_dict=state_dict, model_config=model_config,
                                                         train_config=train_config)
    config['cross_share_all_kv_heads'] = get_config_value(key='cross_share_all_kv_heads', default=True,
                                                          state_dict=state_dict, model_config=model_config,
                                                          train_config=train_config)

    config['seq_attn_isolated'] = get_config_value(key='seq_attn_isolated', default=False, state_dict=state_dict,
                                                   model_config=model_config, train_config=train_config)
    config['seq_attn_serial'] = get_config_value(key='seq_attn_serial', default=False, state_dict=state_dict,
                                                 model_config=model_config, train_config=train_config)

    return config


def build_model(config: dict):
    model = FeaturesTransformer(
        preprocess_config_x=config['preprocess_config_x'],
        encoder_config_x=config['encoder_config_x'],
        encoder_config_y=config['encoder_config_y'],
        decoder_config=config['decoder_config'],
        # feature_positional_embedding_type=config.get('feature_positional_embedding_type', 'subortho'),
        # classify_reg_mixed=config.get('classify_reg_mixed', True),
        feature_positional_embedding_type=config['feature_positional_embedding_type'],
        classify_reg_mixed=config['classify_reg_mixed'],
        nlayers=config['nlayers'],
        nhead=config['nhead'],
        embed_dim=config['embed_dim'],
        hid_dim=config['hid_dim'],
        # mask_feature_embedding_type=config.get('mask_feature_embedding_type', 'mask_embedding'),
        # enable_mask_feature_pred=config.get('enable_mask_feature_pred', True),
        # enable_mask_indicator_pred=config.get('enable_mask_indicator_pred', False),
        mask_feature_embedding_type=config['mask_feature_embedding_type'],
        enable_mask_feature_pred=config['enable_mask_feature_pred'],
        enable_mask_indicator_pred=config['enable_mask_indicator_pred'],
        mask_prediction=config['mask_prediction'],
        features_per_group=config['features_per_group'],
        dropout=config['dropout'],
        # pre_norm=config.get('pre_norm', True),
        pre_norm=config['pre_norm'],
        activation=config.get('activation', 'gelu'),
        layer_norm_eps=config.get('layer_norm_eps', 1e-5),
        device=config.get('device', None),
        dtype=config.get('dtype', None),
        recompute_attn=config['recompute_attn'],
        qkv_w_init_method=config.get('qkv_w_init_method', 'xavier_uniform'),
        mlp_init_std=config.get('mlp_init_std', 0),
        mlp_use_residual=config.get('mlp_use_residual', True),
        layer_arch=config.get('layer_arch', 'fmfmsm'),
        self_share_all_kv_heads=config.get('self_share_all_kv_heads', False),
        cross_share_all_kv_heads=config.get('cross_share_all_kv_heads', False),
        seq_attn_isolated=config.get('seq_attn_isolated', False),
        seq_attn_serial=config.get('seq_attn_serial', False),
    )
    return model


def init_model(model_config: dict, train_config: dict):
    # 1. 保存当前随机状态（用于后续恢复）
    py_rng_state = random.getstate()
    np_rng_state = np.random.get_state()
    torch_rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        cuda_rng_state = torch.cuda.get_rng_state()

    # 2. 设置临时初始化种子
    init_seed = train_config['fixmodelseed']  # 初始化专用种子
    torch.manual_seed(init_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(init_seed)

    config = get_model_config(None, model_config, train_config)
    model = build_model(config)

    random.setstate(py_rng_state)
    np.random.set_state(np_rng_state)
    torch.set_rng_state(torch_rng_state)
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(cuda_rng_state)

    model.to(train_config['device'])
    model.train()
    return model


def load_model(model_path, calculate_sample_attention: bool = False, calculate_feature_attention: bool = False,
               mask_prediction: bool = False):
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    if 'config' in state_dict and 'encoder_config_x' in state_dict['config'] and len(state_dict['config']) > 15:
        config = state_dict['config']
    else:
        config = get_model_config(state_dict, None, None)

    config['calculate_sample_attention'] = calculate_sample_attention
    config['calculate_feature_attention'] = calculate_feature_attention
    config['mask_prediction'] = mask_prediction

    # 构建模型
    model = build_model(config)

    # 模型格式转换
    state_dict['state_dict'] = state_dict_map(state_dict['state_dict'])

    model.load_state_dict(state_dict['state_dict'])

    model.eval()
    return model