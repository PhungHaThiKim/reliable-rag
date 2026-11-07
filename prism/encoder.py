"""Inductive path / subgraph encoder.

Inductive = no entity-ID embedding table. Path representation depends only on
relation types and structural position, so an unseen entity inherits its path
features from its relation neighborhood. Same encoder works on a KG that grows
new nodes after training.
"""
import torch
import torch.nn as nn

from .config import CFG


class PathEncoder(nn.Module):
    """Encodes a path (sequence of relation IDs) into a single vector."""

    def __init__(self, cfg=CFG):
        super().__init__()
        self.cfg = cfg
        self.relation_emb = nn.Embedding(cfg.n_relations, cfg.d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(cfg.max_path_len, cfg.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.d_model * 2,
            dropout=cfg.dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, paths: torch.Tensor) -> torch.Tensor:
        """paths: [N, L] relation ids. 0 = pad. Returns [N, d_model]."""
        n, length = paths.shape
        rel = self.relation_emb(paths)
        pos = self.pos_emb(torch.arange(length, device=paths.device)).unsqueeze(0)
        x = rel + pos
        mask = paths == 0
        h = self.transformer(x, src_key_padding_mask=mask)
        valid = (~mask).unsqueeze(-1).float()
        pooled = (h * valid).sum(1) / valid.sum(1).clamp(min=1)
        return self.out(pooled)


class SubgraphEncoder(nn.Module):
    """Combines path encoder with candidate-level GNN scores."""

    def __init__(self, cfg=CFG):
        super().__init__()
        self.cfg = cfg
        self.path_enc = PathEncoder(cfg)
        self.query_proj = nn.Linear(cfg.d_query, cfg.d_model)
        self.score_proj = nn.Linear(1, 16)
        self.merge = nn.Linear(cfg.d_model + 16, cfg.d_model)

    def forward(self, query_emb: torch.Tensor, paths: torch.Tensor,
                path_to_cand: torch.Tensor, cand_scores: torch.Tensor):
        """
        query_emb:   [d_query]
        paths:       [N, L]
        path_to_cand:[N]
        cand_scores: [C]

        Returns:
          q_h:        [d_model]      query in path space
          path_h:     [N, d_model]   path embeddings (relation-only, inductive)
          cand_score_per_path: [N]   GNN score of each path's candidate
        """
        q_h = self.query_proj(query_emb)
        path_h = self.path_enc(paths)
        cand_score_per_path = cand_scores[path_to_cand]
        score_h = self.score_proj(cand_score_per_path.unsqueeze(-1))
        path_h = self.merge(torch.cat([path_h, score_h], dim=-1))
        return q_h, path_h, cand_score_per_path
