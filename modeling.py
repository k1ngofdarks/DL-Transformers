from __future__ import annotations

import math

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, pad_id: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.embed_dim = embed_dim

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.embed_dim)


class PositionalEncoder(nn.Module):
    def __init__(self, max_length: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        pos_features = torch.zeros(max_length, embed_dim)
        positions = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        freqs = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float) * (-math.log(10000.0) / embed_dim)
        ).unsqueeze(0)

        arguments = positions * freqs
        pos_features[:, 0::2] = torch.sin(arguments)
        pos_features[:, 1::2] = torch.cos(arguments)
        pos_features = pos_features.unsqueeze(0)

        self.register_buffer("pos_features", pos_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pos_features[:, :seq_len, :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        pad_id_src: int,
        pad_id_tgt: int,
        d_model: int = 384,
        nhead: int = 6,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 100,
    ):
        super().__init__()
        self.src_embed = TokenEmbedding(src_vocab, d_model, pad_id=pad_id_src)
        self.tgt_embed = TokenEmbedding(tgt_vocab, d_model, pad_id=pad_id_tgt)
        self.pos_enc = PositionalEncoder(max_len, d_model, dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab)

    def forward(self, src, tgt, pad_id_src: int, pad_id_tgt: int):
        src_key_padding_mask = src == pad_id_src
        tgt_key_padding_mask = tgt == pad_id_tgt
        tgt_len = tgt.size(1)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(tgt.device)

        src_emb = self.pos_enc(self.src_embed(src))
        tgt_emb = self.pos_enc(self.tgt_embed(tgt))

        out = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.fc_out(out)
