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


def load_cls_data(data_root, folder):
    le = LabelEncoder()
    train_path = os.path.join(data_root, folder, folder + '_train.csv')
    test_path = os.path.join(data_root, folder, folder + '_test.csv')
    if os.path.exists(train_path):
        train_df = pd.read_csv(train_path)
        if os.path.exists(test_path):
            test_df = pd.read_csv(test_path)
        else:
            train_df, test_df = train_test_split(train_df, test_size=0.5, random_state=42)
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            try:
                le = LabelEncoder()
                X_train[col] = le.fit_transform(X_train[col])
                X_test[col] = le.transform(X_test[col])
            except Exception as e:
                X_train = X_train.drop(columns=[col])
                X_test = X_test.drop(columns=[col])
