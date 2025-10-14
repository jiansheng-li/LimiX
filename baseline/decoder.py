import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)

        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)

        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Self-attention (masked for autoregressive behavior)
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_output))

        # Cross-attention with encoder output
        cross_attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_output))

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class AdvancedDecoder(nn.Module):
    def __init__(self, d_model=512, num_heads=8, num_layers=6, d_ff=2048,
                 vocab_size=30000, max_len=512, dropout=0.1):
        super(AdvancedDecoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Output layer
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        # tgt: [batch_size, tgt_len]
        # encoder_output: [batch_size, src_len, d_model]

        tgt_embedded = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.pos_encoding(tgt_embedded)
        x = self.dropout(tgt_embedded)

        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        output = self.fc_out(x)
        return output


class CustomDecoder(nn.Module):
    """
    A custom decoder that can be used for post-training with a pre-trained encoder
    """

    def __init__(self, encoder_d_model=512, decoder_config=None):
        super(CustomDecoder, self).__init__()

        # Default configuration
        if decoder_config is None:
            decoder_config = {
                'd_model': encoder_d_model,
                'num_heads': 8,
                'num_layers': 6,
                'd_ff': 2048,
                'vocab_size': 30000,
                'max_len': 512,
                'dropout': 0.1
            }

        self.decoder = AdvancedDecoder(**decoder_config)

        # If we're doing sequence-to-sequence, we might need an adapter
        if encoder_d_model != decoder_config['d_model']:
            self.adapter = nn.Linear(encoder_d_model, decoder_config['d_model'])
        else:
            self.adapter = None

    def forward(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        # Adapt encoder output if needed
        if self.adapter is not None:
            encoder_output = self.adapter(encoder_output)

        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)


# Example usage and integration
class PostTrainingModel(nn.Module):
    def __init__(self, encoder, decoder_config=None):
        super(PostTrainingModel, self).__init__()

        # Assume encoder is already trained and frozen initially
        self.encoder = encoder
        self.decoder = CustomDecoder(
            encoder_d_model=encoder.d_model,  # Assuming encoder has this attribute
            decoder_config=decoder_config
        )

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Get encoder output
        encoder_output = self.encoder(src, mask=src_mask)

        # Decode using custom decoder
        output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)

        return output


# Example instantiation
def create_decoder_for_post_training(encoder_model, decoder_config=None):
    """
    Creates a decoder suitable for post-training with a pre-trained encoder

    Args:
        encoder_model: A pre-trained encoder model
        decoder_config: Configuration dictionary for the decoder

    Returns:
        A model with encoder and new decoder for post-training
    """
    model = PostTrainingModel(encoder_model, decoder_config)
    return model


# Example of how to use the decoder in a training scenario
def example_usage():
    # Example encoder (replace with your actual encoder)
    class DummyEncoder(nn.Module):
        def __init__(self, d_model=512):
            super().__init__()
            self.d_model = d_model
            self.linear = nn.Linear(768, d_model)  # Example transformation

        def forward(self, x, mask=None):
            # Simulate encoder output
            return self.linear(x)

    # Create a dummy encoder (replace with your trained encoder)
    dummy_encoder = DummyEncoder(d_model=512)

    # Define decoder config
    decoder_config = {
        'd_model': 512,
        'num_heads': 8,
        'num_layers': 6,
        'd_ff': 2048,
        'vocab_size': 30000,
        'max_len': 512,
        'dropout': 0.1
    }

    # Create the post-training model
    model = create_decoder_for_post_training(dummy_encoder, decoder_config)

    # Example input
    batch_size = 2
    seq_len = 10
    src = torch.randn(batch_size, seq_len, 768)  # Encoder input
    tgt = torch.randint(0, 30000, (batch_size, seq_len))  # Decoder input

    # Forward pass
    output = model(src, tgt)
    print(f"Output shape: {output.shape}")  # [batch_size, seq_len, vocab_size]

    return model


# Run example
if __name__ == "__main__":
    model = example_usage()
    print("Decoder model created successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")



