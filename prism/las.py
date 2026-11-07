"""Lightweight Answer Simulator (LAS) — heart of PRISM.

Predicts P(answer | query, retrieved_subgraph) over candidate entities,
trained to mimic a frozen LLM via distillation. All four PRISM abilities
(adaptive-K, path importance, calibration, diversity) are derived from this
single distribution.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CFG
from .encoder import SubgraphEncoder


class LAS(nn.Module):
    """Tiny transformer that fuses query + selected paths into a candidate
    distribution. Cheaper than LLM by ~100x for per-step retrieval decisions."""

    def __init__(self, cfg=CFG):
        super().__init__()
        self.cfg = cfg
        self.encoder = SubgraphEncoder(cfg)
        self.cand_proj = nn.Linear(1, cfg.d_model)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout, batch_first=True,
        )
        self.fusion = nn.TransformerEncoder(decoder_layer, num_layers=cfg.n_layers)

        self.cand_score = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, 1),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.ReLU(),
            nn.Linear(cfg.d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, query_emb, paths, path_to_cand, cand_scores,
                selected_path_idx=None):
        """
        Inputs are for ONE sample (no batch). selected_path_idx restricts which
        rows of `paths` form the current S; None = use all paths.

        Returns:
          probs:       [C]   P(answer | q, S) over candidates
          confidence:  scalar (calibrated against actual recall)
          pooled:      [d_model] internal repr (used for online adaptation)
        """
        q_h, path_h, _ = self.encoder(query_emb, paths, path_to_cand, cand_scores)

        if selected_path_idx is not None and len(selected_path_idx) > 0:
            sel_idx = torch.tensor(selected_path_idx, device=paths.device, dtype=torch.long)
            path_h_sel = path_h[sel_idx]
            p2c_sel = path_to_cand[sel_idx]
        elif selected_path_idx is not None and len(selected_path_idx) == 0:
            path_h_sel = path_h.new_zeros(0, path_h.size(-1))
            p2c_sel = path_to_cand.new_zeros(0)
        else:
            path_h_sel = path_h
            p2c_sel = path_to_cand

        n_cand = cand_scores.size(0)
        cand_token = self.cand_proj(cand_scores.unsqueeze(-1))   # [C, d]

        tokens = torch.cat([q_h.unsqueeze(0), cand_token, path_h_sel], dim=0)
        h = self.fusion(tokens.unsqueeze(0)).squeeze(0)            # [1+C+|S|, d]

        cand_h = h[1:1 + n_cand]
        cand_logits = self.cand_score(cand_h).squeeze(-1)
        probs = F.softmax(cand_logits, dim=0)

        pooled = h.mean(0)
        confidence = self.confidence_head(pooled).squeeze(-1)
        return probs, confidence, pooled

    def kl_to_teacher(self, probs: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        eps = 1e-9
        return (teacher * (teacher.clamp(min=eps).log() - probs.clamp(min=eps).log())).sum()

    # ─────────────────────────────────────────────────────────────────
    # Batched helpers — exposed for use by controller_batched.py
    # ─────────────────────────────────────────────────────────────────

    def encode_once(self, query_emb, paths, path_to_cand, cand_scores):
        """Run encoder + cand projection ONCE per query.

        Returns reusable tensors so the controller doesn't re-encode paths
        for every counterfactual trial.

        Returns:
            q_h:        [d]              query token
            path_h:     [N_total, d]     all-path embeddings (after merge)
            cand_token: [C, d]           candidate tokens
        """
        q_h, path_h, _ = self.encoder(query_emb, paths, path_to_cand, cand_scores)
        cand_token = self.cand_proj(cand_scores.unsqueeze(-1))   # [C, d]
        return q_h, path_h, cand_token

    def fuse_subset(self, q_h, cand_token, path_h_subset):
        """Single-subset fusion. Same logic as forward(...) but takes
        pre-computed tensors. Useful for inference WITHOUT re-encoding.

        Returns: (probs[C], confidence, pooled[d])
        """
        import torch.nn.functional as F
        n_cand = cand_token.size(0)
        tokens = torch.cat([q_h.unsqueeze(0), cand_token, path_h_subset], dim=0)
        h = self.fusion(tokens.unsqueeze(0)).squeeze(0)              # [1+C+|S|, d]
        cand_h = h[1:1 + n_cand]
        cand_logits = self.cand_score(cand_h).squeeze(-1)
        probs = F.softmax(cand_logits, dim=0)
        pooled = h.mean(0)
        confidence = self.confidence_head(pooled).squeeze(-1)
        return probs, confidence, pooled

    def fuse_batched_trials(self, q_h, cand_token, path_h_S, path_h_trials):
        """Batched fusion for K_max counterfactual trials simultaneously.

        For each trial path p_i, computes P_phi(a | q, S ∪ {p_i}) in ONE
        forward pass. ~100x faster than calling fuse_subset in a loop.

        Args:
            q_h:           [d]            query token
            cand_token:    [C, d]         candidate tokens
            path_h_S:      [|S|, d]       embeddings of paths currently in S
                                          (use new_zeros(0, d) for empty S)
            path_h_trials: [N_trial, d]   embeddings of candidate trial paths

        Returns:
            probs:      [N_trial, C]      one distribution per trial
            confidence: [N_trial]         one confidence per trial
        """
        import torch.nn.functional as F
        n_trial = path_h_trials.size(0)
        n_cand = cand_token.size(0)
        d = q_h.size(0)

        if n_trial == 0:
            return (torch.zeros(0, n_cand, device=q_h.device),
                    torch.zeros(0, device=q_h.device))

        # Base = [q, cand_1, ..., cand_C, S_path_1, ..., S_path_|S|]   shape [B, d]
        base = torch.cat([q_h.unsqueeze(0), cand_token, path_h_S], dim=0)  # [B, d]
        # Replicate base across N_trial, append each trial path
        base_rep = base.unsqueeze(0).expand(n_trial, -1, -1)               # [N_trial, B, d]
        trial_tok = path_h_trials.unsqueeze(1)                              # [N_trial, 1, d]
        tokens = torch.cat([base_rep, trial_tok], dim=1)                    # [N_trial, B+1, d]

        h = self.fusion(tokens)                                             # [N_trial, B+1, d]
        cand_h = h[:, 1:1 + n_cand]                                         # [N_trial, C, d]
        cand_logits = self.cand_score(cand_h).squeeze(-1)                   # [N_trial, C]
        probs = F.softmax(cand_logits, dim=-1)
        pooled = h.mean(dim=1)                                              # [N_trial, d]
        confidence = self.confidence_head(pooled).squeeze(-1)               # [N_trial]
        return probs, confidence
