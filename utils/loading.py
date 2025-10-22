import json

import torch

from model.transformer import FeaturesTransformer


def load_model(model_path, mask_prediction: bool = False):
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    config = state_dict['config']
    model = FeaturesTransformer(
        preprocess_config_x=config['preprocess_config_x'],
        encoder_config_x=config['encoder_config_x'],
        encoder_config_y=config['encoder_config_y'],
        decoder_config=config['decoder_config'],
        nlayers=config['nlayers'],
        nhead=config['nhead'],
        embed_dim=config['embed_dim'],
        hid_dim=config['hid_dim'],
        mask_prediction=mask_prediction,
        features_per_group=config['features_per_group'],
        dropout=config['dropout'],
        layer_norm_eps=config.get('layer_norm_eps', 1e-5),
        device=None,
        dtype=None,
        recompute_attn=config['recompute_attn']
    )
    model.load_state_dict(state_dict['state_dict'])
    model.eval()
    return model


def load_finetuning_model(model_path,config_path="model_dict.json"):
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    config = state_dict['config']
    model = FeaturesTransformer(
        preprocess_config_x=config['preprocess_config_x'],
        encoder_config_x=config['encoder_config_x'],
        encoder_config_y=config['encoder_config_y'],
        decoder_config=config['decoder_config'],
        nlayers=config['nlayers'],
        nhead=config['nhead'],
        embed_dim=config['embed_dim'],
        hid_dim=config['hid_dim'],
        mask_prediction=False,
        features_per_group=config['features_per_group'],
        dropout=config['dropout'],
        layer_norm_eps=config.get('layer_norm_eps', 1e-5),
        device=None,
        dtype=None,
        recompute_attn=config['recompute_attn']
    )
    model.load_state_dict(state_dict['state_dict'])
    model_dict=json.load(open("model_dict.json","r"))
    for key in model_dict.keys():
        if not model_dict[key]:
            model[key].requires_grad=False
    model.train()
    return model
