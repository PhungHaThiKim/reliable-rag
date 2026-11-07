"""Method improvements layered on top of PRISM's base RL training.

Each improvement is a drop-in replacement or extension for a specific piece
of the training pipeline. Use individually or compose. All are paper-relevant
contributions usable as ablation rows.

Improvements:
    1. CalibrationShapedReward       — penalise overconfidence / underconfidence
    2. DifficultyAwareReward         — adapt α weights per question difficulty
    3. EntropyRegularizedRL          — add policy entropy bonus
    4. SelectiveAbstentionEval       — risk-coverage curve via confidence
    5. OnlineAdaptation              — closed-loop deployment update
    6. TemperatureCalibration        — post-hoc Platt-style temperature scaling
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CFG
from .controller import ControllerTrace
from .data import Sample
from .las import LAS
from .metrics import coverage as cov_metric, diversity as div_metric


# ─────────────────────────────────────────────────────────────────────
# 1. Calibration-shaped reward — penalise (confidence, accuracy) mismatch
# ─────────────────────────────────────────────────────────────────────

@dataclass
class CalibShapeCfg:
    overconf_weight: float = 0.2      # penalty for high conf + wrong answer
    underconf_weight: float = 0.1     # penalty for low conf + right answer
    overconf_thr: float = 0.7
    underconf_thr: float = 0.3


def calib_shaped_reward(sample: Sample, selected: List[int],
                          confidence: float, cfg=CFG,
                          shape: CalibShapeCfg = CalibShapeCfg()) -> float:
    """Augmented reward: cov + eff + div − overconf_penalty − underconf_penalty.

    Tightens the link between predicted confidence and actual recall.
    Theorem (informal): with this shaped reward, optimal policy must produce
    confidence ≈ E[recall] → calibration is in the loss, not a post-hoc fix.
    """
    cov = cov_metric(selected, sample.path_to_candidate,
                     sample.candidate_ids, sample.gold_entities)
    eff = 1.0 - len(selected) / max(1, cfg.max_select_paths)
    div = div_metric(selected, sample.path_to_candidate)
    base = (cfg.rl_w_coverage * cov +
            cfg.rl_w_efficiency * eff +
            cfg.rl_w_diversity * div)

    # Shape: discourage confident-wrong and unconfident-right
    if cov < 0.5 and confidence > shape.overconf_thr:
        base -= shape.overconf_weight * (confidence - shape.overconf_thr)
    if cov > 0.8 and confidence < shape.underconf_thr:
        base -= shape.underconf_weight * (shape.underconf_thr - confidence)

    return base


# ─────────────────────────────────────────────────────────────────────
# 2. Difficulty-aware reward — adapt α weights per question type
# ─────────────────────────────────────────────────────────────────────

def _estimate_difficulty(sample: Sample) -> str:
    """Cheap proxy: long paths or many candidates → hard."""
    avg_path_len = (sample.paths != 0).any(dim=-1).sum().item()  # # non-empty
    n_cand = sample.candidate_ids.size(0)
    n_paths = sample.paths.size(0)
    if n_paths < 5 or n_cand <= 5:
        return "easy"
    if n_paths > 30 or avg_path_len > 25:
        return "hard"
    return "medium"


def difficulty_aware_reward(sample: Sample, selected: List[int],
                              cfg=CFG) -> float:
    """Weight coverage HIGHER on hard questions (we need more evidence) and
    efficiency HIGHER on easy questions (don't waste budget)."""
    diff = _estimate_difficulty(sample)
    if diff == "easy":
        w_cov, w_eff, w_div = 0.4, 0.4, 0.2
    elif diff == "hard":
        w_cov, w_eff, w_div = 0.8, 0.1, 0.1
    else:
        w_cov, w_eff, w_div = cfg.rl_w_coverage, cfg.rl_w_efficiency, cfg.rl_w_diversity

    cov = cov_metric(selected, sample.path_to_candidate,
                     sample.candidate_ids, sample.gold_entities)
    eff = 1.0 - len(selected) / max(1, cfg.max_select_paths)
    div = div_metric(selected, sample.path_to_candidate)
    return w_cov * cov + w_eff * eff + w_div * div


# ─────────────────────────────────────────────────────────────────────
# 3. Entropy-regularised RL — explicit exploration bonus
# ─────────────────────────────────────────────────────────────────────

def rl_step_entropy_reg(las: LAS, sample: Sample,
                          reward_fn: Callable[[Sample, List[int], float], float],
                          cfg=CFG, entropy_coef: float = 0.01,
                          use_batched: bool = True):
    """Variant of rl_step with entropy bonus and pluggable reward.

    Returns: (loss, mean_reward, baseline_reward, entropy_mean)

    Use ANY of the reward variants:
        reward_fn = lambda s,sel,conf: calib_shaped_reward(s, sel, conf)
        reward_fn = lambda s,sel,conf: difficulty_aware_reward(s, sel)
    """
    if use_batched:
        from .controller_batched import deterministic_select, stochastic_select
    else:
        from .controller import deterministic_select, stochastic_select

    with torch.no_grad():
        baseline_trace = deterministic_select(las, sample, cfg)
        baseline_conf = float(baseline_trace.confidence.item())
        baseline_reward = reward_fn(sample, baseline_trace.selected_path_idx,
                                      baseline_conf)

    pg_loss = 0.0
    calib_loss = 0.0
    entropy_acc = 0.0
    rewards: List[float] = []

    for _ in range(cfg.rl_n_samples):
        trace = stochastic_select(las, sample, cfg)
        conf_s = float(trace.confidence.item())
        r = reward_fn(sample, trace.selected_path_idx, conf_s)
        rewards.append(r)
        advantage = r - baseline_reward

        if trace.log_probs:
            log_p_sum = torch.stack(trace.log_probs).sum()
            pg_loss = pg_loss + (-advantage * log_p_sum)
            # BUGFIX: the old "entropy proxy" was `(-log_p_sum).detach()`,
            # i.e. a constant w.r.t. the parameters — the entropy term never
            # produced a gradient. Use the true (differentiable) policy
            # entropy recorded by the stochastic controller.
            if getattr(trace, "entropies", None):
                entropy_acc = entropy_acc + torch.stack(trace.entropies).mean()
            else:  # legacy controller without entropies → no bonus
                entropy_acc = entropy_acc + torch.zeros((), device=log_p_sum.device)

        # ── Issue #3 fix: calibrate against ACTUAL recall (set overlap),
        # not student's expected recall. trace.final_probs is the student;
        # using it as target would be self-consistency, not calibration.
        # Since RL does not see teacher_probs per step, use actual gold-set
        # coverage from the selected paths as the empirical recall target.
        gold_set = set(sample.gold_entities.tolist())
        # Actual recall = fraction of gold entities the controller touched
        touched = set()
        for idx in trace.selected_path_idx:
            cand = int(sample.path_to_candidate[idx].item())
            touched.add(int(sample.candidate_ids[cand].item()))
        actual_recall = len(touched & gold_set) / max(len(gold_set), 1)
        calib_loss = calib_loss + (trace.confidence - actual_recall) ** 2

    n = max(1, cfg.rl_n_samples)
    # Entropy regularisation: subtract entropy proxy → encourage exploration
    entropy_term = entropy_acc / n
    total = (pg_loss / n
             + cfg.rl_calib_weight * calib_loss / n
             - entropy_coef * entropy_term)

    ent_val = float(entropy_term.detach().item()) if isinstance(entropy_term, torch.Tensor) else float(entropy_term)
    return total, sum(rewards) / n, baseline_reward, ent_val


# ─────────────────────────────────────────────────────────────────────
# 4. Selective abstention — risk-coverage curve evaluation
# ─────────────────────────────────────────────────────────────────────

def risk_coverage_curve(confidences: Sequence[float],
                         correctness: Sequence[int],
                         n_points: int = 20) -> List[Tuple[float, float]]:
    """Plot data: x = coverage (% answered), y = accuracy on answered subset.

    A well-calibrated retriever produces a monotonically decreasing curve as
    we lower the confidence threshold (answer more, accept lower confidence).
    Compare PRISM vs baselines: steeper curve = more useful confidence.
    """
    if not confidences:
        return []
    pairs = sorted(zip(confidences, correctness), key=lambda x: -x[0])
    n = len(pairs)
    out: List[Tuple[float, float]] = []
    step = max(1, n // n_points)
    cum_correct = 0
    for i, (_, ok) in enumerate(pairs, 1):
        cum_correct += int(ok)
        if i % step == 0 or i == n:
            coverage_frac = i / n
            accuracy = cum_correct / i
            out.append((coverage_frac, accuracy))
    return out


def selective_abstention_eval(confidences: Sequence[float],
                                correctness: Sequence[int],
                                abstain_thresholds: Sequence[float] =
                                (0.3, 0.5, 0.7, 0.9)):
    """Headline numbers for abstention analysis."""
    confs = list(confidences)
    corrs = list(correctness)
    n = len(confs)
    out = []
    for thr in abstain_thresholds:
        keep_idx = [i for i, c in enumerate(confs) if c >= thr]
        if not keep_idx:
            out.append({"threshold": thr, "coverage": 0.0, "hit": 0.0})
            continue
        kept_corr = [corrs[i] for i in keep_idx]
        out.append({
            "threshold": thr,
            "coverage": len(keep_idx) / n,
            "hit": sum(kept_corr) / len(kept_corr),
        })
    return out


# ─────────────────────────────────────────────────────────────────────
# 5. Online closed-loop adaptation — deployment-time update
# ─────────────────────────────────────────────────────────────────────

class OnlineAdapter:
    """Maintains a small buffer of (sample, observed_probs) and periodically
    runs k SGD steps on a teacher-distill loss to keep LAS aligned with the
    LLM as deployment data shifts.

    Usage:
        adapter = OnlineAdapter(las, lr=1e-5, buffer_size=50, update_steps=5)
        for q in deployment_stream:
            trace = controller.deterministic_select(las, q)
            actual_probs = real_llm_score(q, trace.selected_path_idx)
            adapter.observe(q, trace.selected_path_idx, actual_probs)
            if adapter.should_update():
                adapter.step()
    """

    def __init__(self, las: LAS, lr: float = 1e-5,
                 buffer_size: int = 50, update_steps: int = 5,
                 max_buffer_keep: int = 200):
        self.las = las
        self.buffer: List[Tuple[Sample, List[int], torch.Tensor]] = []
        self.buffer_size = buffer_size
        self.update_steps = update_steps
        self.max_buffer_keep = max_buffer_keep
        self.opt = torch.optim.AdamW(las.parameters(), lr=lr, weight_decay=1e-5)

    def observe(self, sample: Sample, selected: List[int],
                 actual_probs: torch.Tensor):
        """Add one deployment observation."""
        self.buffer.append((sample, list(selected), actual_probs.detach().cpu()))
        if len(self.buffer) > self.max_buffer_keep:
            self.buffer.pop(0)

    def should_update(self) -> bool:
        return len(self.buffer) >= self.buffer_size

    def step(self):
        """Take `update_steps` SGD steps on accumulated buffer."""
        if not self.buffer:
            return {"loss": 0.0, "n_samples": 0}
        self.las.train()
        device = next(self.las.parameters()).device

        recent = self.buffer[-self.buffer_size:]
        total_loss = 0.0
        for _ in range(self.update_steps):
            self.opt.zero_grad()
            batch_loss = 0.0
            for sample, selected, target_probs in recent:
                sample.query_emb = sample.query_emb.to(device)
                sample.paths = sample.paths.to(device)
                sample.path_to_candidate = sample.path_to_candidate.to(device)
                sample.candidate_gnn_scores = sample.candidate_gnn_scores.to(device)
                target = target_probs.to(device)

                probs, _, _ = self.las(
                    sample.query_emb, sample.paths,
                    sample.path_to_candidate, sample.candidate_gnn_scores,
                    selected_path_idx=selected,
                )
                if probs.shape != target.shape:
                    continue
                loss = self.las.kl_to_teacher(probs, target)
                loss.backward()
                batch_loss += float(loss.item())
            torch.nn.utils.clip_grad_norm_(self.las.parameters(), 0.5)
            self.opt.step()
            total_loss += batch_loss / max(len(recent), 1)
        self.las.eval()
        return {"loss": total_loss / self.update_steps, "n_samples": len(recent)}


# ─────────────────────────────────────────────────────────────────────
# 6. Temperature calibration (post-hoc Platt scaling for confidence)
# ─────────────────────────────────────────────────────────────────────

class TemperatureScaler(nn.Module):
    """Post-hoc calibration. Fits a single scalar T on the dev set so that
    σ(z/T) is well-calibrated.

    Usage:
        ts = TemperatureScaler().to(device)
        ts.fit(confidences_dev, correctness_dev)
        calibrated_conf = ts(raw_conf)
    """

    def __init__(self):
        super().__init__()
        self.log_T = nn.Parameter(torch.zeros(1))   # T = exp(log_T)

    @property
    def T(self) -> torch.Tensor:
        return self.log_T.exp()

    def forward(self, conf: torch.Tensor) -> torch.Tensor:
        # conf is already in [0,1]; convert to logit, divide by T, back to prob
        eps = 1e-6
        logit = torch.log(conf.clamp(eps, 1 - eps) / (1 - conf).clamp(eps, 1 - eps))
        return torch.sigmoid(logit / self.T)

    def fit(self, raw_confidences: Sequence[float],
            correctness: Sequence[int],
            n_iter: int = 200, lr: float = 0.05) -> float:
        device = self.log_T.device
        conf = torch.tensor(raw_confidences, dtype=torch.float32, device=device)
        y = torch.tensor(correctness, dtype=torch.float32, device=device)
        opt = torch.optim.LBFGS([self.log_T], lr=lr, max_iter=n_iter)

        def closure():
            opt.zero_grad()
            p = self.forward(conf)
            eps = 1e-6
            loss = -(y * p.clamp(eps).log() +
                     (1 - y) * (1 - p).clamp(eps).log()).mean()
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            p = self.forward(conf)
            eps = 1e-6
            final_nll = -(y * p.clamp(eps).log() +
                           (1 - y) * (1 - p).clamp(eps).log()).mean()
        return float(final_nll.item())


# ─────────────────────────────────────────────────────────────────────
# Convenience: full RL train loop with all improvements enabled
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_reward(las: LAS, eval_set, reward_fn, cfg=CFG,
                    use_batched: bool = True, max_items: Optional[int] = None) -> float:
    """Mean greedy-controller reward on eval_set (model-selection signal)."""
    if use_batched:
        from .controller_batched import deterministic_select
    else:
        from .controller import deterministic_select
    las.eval()
    device = next(las.parameters()).device
    total, n = 0.0, 0
    for i in range(len(eval_set) if max_items is None else min(max_items, len(eval_set))):
        sample = eval_set[i]
        for f in ["query_emb", "paths", "path_to_candidate",
                  "candidate_gnn_scores", "candidate_ids", "gold_entities"]:
            setattr(sample, f, getattr(sample, f).to(device))
        trace = deterministic_select(las, sample, cfg)
        total += reward_fn(sample, trace.selected_path_idx, float(trace.confidence.item()))
        n += 1
    las.train()
    return total / max(n, 1)


def train_rl_improved(las: LAS, train_set, eval_set, cfg=CFG,
                       reward_mode: str = "calib_shaped",
                       entropy_coef: float = 0.02,
                       use_batched: bool = True,
                       eval_every: int = 50,
                       log_every: int = 25,
                       patience: Optional[int] = None,
                       eval_max_items: Optional[int] = None):
    """Drop-in replacement for rl.train_rl with improvements applied.

    reward_mode: "base" | "calib_shaped" | "difficulty"

    BUGFIX vs. the original: the best checkpoint is now chosen by the
    greedy reward on `eval_set` (previously: the exponential running
    TRAIN reward, which both ignores the eval set and is biased by the
    stochastic samples). `patience` (in eval rounds) enables early stop;
    None = cfg.rl_patience.
    """
    import os
    from tqdm import tqdm

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    las.to(device)
    opt = torch.optim.AdamW(las.parameters(), lr=cfg.rl_lr, weight_decay=1e-4)

    if reward_mode == "calib_shaped":
        reward_fn = lambda s, sel, conf: calib_shaped_reward(s, sel, conf, cfg)
    elif reward_mode == "difficulty":
        reward_fn = lambda s, sel, conf: difficulty_aware_reward(s, sel, cfg)
    elif reward_mode == "base":
        reward_fn = lambda s, sel, conf: (
            cfg.rl_w_coverage * cov_metric(sel, s.path_to_candidate,
                                            s.candidate_ids, s.gold_entities) +
            cfg.rl_w_efficiency * (1.0 - len(sel) / max(1, cfg.max_select_paths)) +
            cfg.rl_w_diversity * div_metric(sel, s.path_to_candidate)
        )
    else:
        raise ValueError(f"unknown reward_mode={reward_mode}")

    os.makedirs(cfg.out_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.out_dir, "prism_rl_improved_best.pt")
    patience = patience if patience is not None else getattr(cfg, "rl_patience", 3)
    min_steps = getattr(cfg, "rl_min_steps", 0)
    min_delta = getattr(cfg, "rl_min_delta", 1e-3)

    # Step-0 reference: the Stage-1 checkpoint is a valid candidate too.
    best_reward = evaluate_reward(las, eval_set, reward_fn, cfg, use_batched, eval_max_items)
    torch.save(las.state_dict(), ckpt_path)
    evals_since_improvement = 0
    history = [{"step": 0, "eval_reward": best_reward}]

    pbar = tqdm(range(cfg.rl_steps), desc="rl-improved")
    running_r, running_b, running_h = 0.0, 0.0, 0.0
    for step in pbar:
        sample = train_set[step % len(train_set)]
        sample.query_emb = sample.query_emb.to(device)
        sample.paths = sample.paths.to(device)
        sample.path_to_candidate = sample.path_to_candidate.to(device)
        sample.candidate_gnn_scores = sample.candidate_gnn_scores.to(device)
        sample.candidate_ids = sample.candidate_ids.to(device)
        sample.gold_entities = sample.gold_entities.to(device)

        opt.zero_grad()
        loss, mean_r, base_r, ent = rl_step_entropy_reg(
            las, sample, reward_fn, cfg=cfg,
            entropy_coef=entropy_coef, use_batched=use_batched,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(las.parameters(), cfg.rl_clip_norm)
        opt.step()

        running_r = 0.95 * running_r + 0.05 * mean_r
        running_b = 0.95 * running_b + 0.05 * base_r
        running_h = 0.95 * running_h + 0.05 * ent

        if step % log_every == 0:
            pbar.set_postfix(R=f"{running_r:.3f}", B=f"{running_b:.3f}",
                              H=f"{running_h:.2f}",
                              adv=f"{running_r - running_b:+.3f}")

        if step > 0 and step % eval_every == 0:
            eval_r = evaluate_reward(las, eval_set, reward_fn, cfg, use_batched, eval_max_items)
            history.append({"step": step, "eval_reward": eval_r,
                            "train_running_reward": running_r})
            if eval_r > best_reward + min_delta:
                best_reward = eval_r
                torch.save(las.state_dict(), ckpt_path)
                evals_since_improvement = 0
                pbar.write(f"[rl-improved] step {step}  eval_R={eval_r:.4f}  ↓ saved best")
            else:
                evals_since_improvement += 1
                pbar.write(f"[rl-improved] step {step}  eval_R={eval_r:.4f}  "
                            f"best={best_reward:.4f}  (no improvement "
                            f"{evals_since_improvement}/{patience})")
                if step >= min_steps and evals_since_improvement >= patience:
                    pbar.write(f"[rl-improved] early stop at step {step}")
                    break

    train_rl_improved.last_history = history
    return ckpt_path
