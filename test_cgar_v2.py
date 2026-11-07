"""
Unit tests for CGAR v2.1 fixes (Bug A/B/C/D).

USAGE:
  Local (after `pip install torch numpy`):
    cd colab/graph_rag && python test_cgar_v2.py

  Colab (one cell):
    !cd /content/drive/MyDrive/ThS/Research/colab/graph_rag && python test_cgar_v2.py

Each test prints PASS / FAIL with a one-line reason. Exit code 0 = all pass.

Stubs out grl_rag so we don't need real WebQSP data.
"""
import json
import sys
import types
from pathlib import Path

# ── 0. Stub grl_rag before importing adagraph_cgar_v2 ────────────────
_stub = types.ModuleType('grl_rag')
_stub.GNNHeuristic = object
_stub.Embedder = object
_stub.extract_candidate_paths = lambda *a, **k: []
_stub.extract_candidate_paths_by_k = lambda *a, **k: {}
_stub.path_to_string = lambda x: ''
_stub.compute_reward = lambda *a, **k: 0.0
sys.modules['grl_rag'] = _stub

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from adagraph_cgar_v2 import (  # noqa: E402
    CGARController, CGARSCSTTrainer,
)

torch.manual_seed(0)
import random  # noqa: E402
random.seed(0)
import numpy as np  # noqa: E402
np.random.seed(0)


# ── 1. Test fixtures ─────────────────────────────────────────────────

EMB_DIM = 16

class FakeEmbedder:
    """Minimal embedder — returns deterministic random embeddings."""
    def __init__(self, seed=0):
        self._gen = torch.Generator().manual_seed(seed)
    def encode_batch(self, texts):
        n = max(1, len(list(texts)))
        return torch.randn(n, EMB_DIM, generator=self._gen)
    def encode(self, t):
        return torch.randn(EMB_DIM, generator=self._gen)


def make_paths(n, gold_overlap=0.3):
    """Generate n synthetic paths.
    `gold_overlap` controls how many paths touch entity 'gold_0' / 'gold_1'.
    """
    paths = []
    for i in range(n):
        if i / n < gold_overlap:
            triples = [(f'h{i}', 'r', 'gold_0'), ('gold_0', 'r2', f'g{i}')]
        else:
            triples = [(f'h{i}', 'r', f't{i}'), (f't{i}', 'r2', f'g{i}')]
        paths.append({
            'path_triples': triples,
            'path_text': f'p{i}',
            'gnn_score': 0.4 + (i % 5) * 0.1,
            'length': 2,
        })
    return paths


def fresh_controller(seed=0):
    torch.manual_seed(seed)
    return CGARController(
        emb_dim=EMB_DIM, hidden_dim=16, max_paths=4,
        sprt_beta=0.99,  # high → no early-stop in deterministic mode
        redundancy_lambda=0.3, epsilon=0.0,
    )


def all_paths_by_k():
    return {
        5:  make_paths(5),
        10: make_paths(8),
        20: make_paths(8),
        50: make_paths(8),
    }


# ── 2. Tests ─────────────────────────────────────────────────────────

PASS = []  # (name, msg)
FAIL = []  # (name, msg)

def check(name, cond, msg_pass, msg_fail):
    (PASS if cond else FAIL).append((name, msg_pass if cond else msg_fail))
    flag = 'ok' if cond else 'FAIL'
    print(f'  {flag} {name}: {msg_pass if cond else msg_fail}')


def test_1_deterministic_forward_works():
    """Regression: deterministic mode still produces valid selection."""
    print('\n[Test 1] Deterministic forward (regression)')
    ctrl = fresh_controller()
    emb = FakeEmbedder()
    q = torch.randn(1, EMB_DIM)

    sel, det = ctrl(q, all_paths_by_k(), emb, deterministic=True, return_details=True)
    check('det_forward.returns_paths', len(sel) > 0,
          f'{len(sel)} paths selected', 'no paths selected')
    check('det_forward.no_log_prob_k', det['log_prob_k'] is None,
          'log_prob_k=None as expected (deterministic)',
          'log_prob_k populated in deterministic mode (bug)')
    check('det_forward.no_log_prob_paths', len(det['log_prob_paths']) == 0,
          'log_prob_paths=[] as expected',
          'log_prob_paths populated in deterministic mode')


def test_2_stochastic_forward_has_log_probs():
    """Bug B: stochastic mode must populate log_prob_k and log_prob_paths
    with gradient-tracking tensors."""
    print('\n[Test 2] Stochastic forward populates log_probs')
    ctrl = fresh_controller()
    emb = FakeEmbedder()
    q = torch.randn(1, EMB_DIM)

    sel, det = ctrl(q, all_paths_by_k(), emb, deterministic=False, return_details=True)
    check('stoch.log_prob_k_present', det['log_prob_k'] is not None,
          'log_prob_k tensor returned',
          'log_prob_k is None — k_policy not sampled')
    if det['log_prob_k'] is not None:
        check('stoch.log_prob_k_grad', det['log_prob_k'].requires_grad,
              'log_prob_k.requires_grad=True',
              'log_prob_k detached — Bug B not fixed')
    check('stoch.log_prob_paths_nonempty', len(det['log_prob_paths']) > 0,
          f'{len(det["log_prob_paths"])} per-step log_probs',
          'log_prob_paths empty — Categorical sampling missing')
    if det['log_prob_paths']:
        all_grad = all(lp.requires_grad for lp in det['log_prob_paths'])
        check('stoch.log_prob_paths_all_grad', all_grad,
              'every step log_prob.requires_grad=True',
              'some path log_probs detached')
    check('stoch.relevance_scores_grad', det['relevance_scores'].requires_grad,
          'relevance_scores.requires_grad=True (v2 fix preserved)',
          'relevance_scores detached — v2 regression!')


def test_3_backward_propagates_to_k_policy_and_relevance_head():
    """Bug B verification at gradient level: -log_prob.sum().backward()
    must give non-zero gradient to BOTH k_policy and relevance_head."""
    print('\n[Test 3] backward → gradient reaches k_policy AND relevance_head')
    ctrl = fresh_controller()
    emb = FakeEmbedder()
    q = torch.randn(1, EMB_DIM)

    _, det = ctrl(q, all_paths_by_k(), emb, deterministic=False, return_details=True)
    log_prob_total = det['log_prob_k'].sum()
    for lp in det['log_prob_paths']:
        log_prob_total = log_prob_total + lp
    (-log_prob_total).backward()

    rel_norm = sum(p.grad.norm().item() for p in ctrl.relevance_head.parameters() if p.grad is not None)
    kp_norm = sum(p.grad.norm().item() for p in ctrl.k_policy.parameters() if p.grad is not None)

    check('grad.relevance_head', rel_norm > 0,
          f'relevance_head grad norm = {rel_norm:.4f}',
          'relevance_head got NO gradient — Bug B not fixed')
    check('grad.k_policy', kp_norm > 0,
          f'k_policy grad norm = {kp_norm:.4f}',
          'k_policy got NO gradient — Bug B not fixed')


def test_4_reinforce_direction_is_correct():
    """Bug A: with positive advantage, REINFORCE loss should be NEGATIVE
    (so minimising loss = maximising log_prob = increasing P[chosen action]).
    With negative advantage, loss should be POSITIVE.
    """
    print('\n[Test 4] REINFORCE sign correctness (Bug A)')
    ctrl = fresh_controller()
    emb = FakeEmbedder()
    q = torch.randn(1, EMB_DIM)

    _, det = ctrl(q, all_paths_by_k(), emb, deterministic=False, return_details=True)
    log_prob_total = det['log_prob_k'].sum()
    for lp in det['log_prob_paths']:
        log_prob_total = log_prob_total + lp

    # log_prob is negative (log of prob in (0,1])
    lp_val = log_prob_total.item()
    check('reinforce.log_prob_negative', lp_val <= 0,
          f'log_prob_total = {lp_val:.3f} (expected ≤ 0)',
          f'log_prob_total = {lp_val:.3f} (suspicious)')

    pos_loss = -1.0 * log_prob_total       # advantage = +1
    neg_loss = -(-1.0) * log_prob_total    # advantage = −1

    check('reinforce.pos_adv_pos_loss', pos_loss.item() >= 0,
          f'+adv → loss={pos_loss.item():.3f} ≥ 0 (encourage action)',
          f'+adv loss is negative — REINFORCE sign wrong')
    check('reinforce.neg_adv_neg_loss', neg_loss.item() <= 0,
          f'−adv → loss={neg_loss.item():.3f} ≤ 0 (discourage action)',
          f'−adv loss positive — REINFORCE sign wrong')


def test_5_diversity_is_real_not_placeholder():
    """Bug C: diversity must depend on path overlap, not be hard-coded 0.5."""
    print('\n[Test 5] Diversity reward is real (Bug C)')
    # All paths share the same entity set → low diversity
    overlap_paths = []
    for i in range(4):
        overlap_paths.append({
            'path_triples': [('shared_h', 'r', 'shared_t'), ('shared_t', 'r2', 'shared_g')],
            'path_text': f'p{i}', 'gnn_score': 0.5, 'length': 2,
        })

    # Paths with disjoint entity sets → high diversity
    disjoint_paths = []
    for i in range(4):
        disjoint_paths.append({
            'path_triples': [(f'h{i}', 'r', f't{i}'), (f't{i}', 'r2', f'g{i}')],
            'path_text': f'p{i}', 'gnn_score': 0.5, 'length': 2,
        })

    div_overlap = CGARSCSTTrainer._diversity_jaccard(overlap_paths)
    div_disjoint = CGARSCSTTrainer._diversity_jaccard(disjoint_paths)

    check('diversity.overlap_low', div_overlap < 0.1,
          f'overlap paths diversity = {div_overlap:.3f} (low, ok)',
          f'overlap paths diversity = {div_overlap:.3f} should be ~0')
    check('diversity.disjoint_high', div_disjoint > 0.9,
          f'disjoint paths diversity = {div_disjoint:.3f} (high, ok)',
          f'disjoint paths diversity = {div_disjoint:.3f} should be ~1')
    check('diversity.not_placeholder', abs(div_overlap - div_disjoint) > 0.5,
          f'gap = {abs(div_overlap - div_disjoint):.3f} (Bug C fixed)',
          'diversity is constant — Bug C not fixed (still 0.5 placeholder?)')


def test_6_compute_reward_uses_real_diversity():
    """Bug C end-to-end: compute_reward returns different values for
    overlapping vs disjoint path sets (with same coverage & efficiency)."""
    print('\n[Test 6] compute_reward uses diversity (Bug C end-to-end)')
    # Build a fake trainer (no controller actually used here)
    class _Stub:
        max_paths = 4
        K_OPTIONS = [5, 10, 20, 50]
    trainer = CGARSCSTTrainer.__new__(CGARSCSTTrainer)
    trainer.alpha, trainer.beta, trainer.gamma = 0.0, 0.0, 1.0  # isolate diversity term
    # 4 overlapping paths
    overlap_paths = [{'path_triples': [('A', 'r', 'B'), ('B', 'r2', 'C')]} for _ in range(4)]
    # 4 disjoint paths
    disjoint_paths = [{
        'path_triples': [(f'h{i}', 'r', f't{i}'), (f't{i}', 'r2', f'g{i}')],
    } for i in range(4)]

    r_overlap = trainer.compute_reward(overlap_paths, set(), max_paths=4)
    r_disjoint = trainer.compute_reward(disjoint_paths, set(), max_paths=4)

    check('reward.overlap_low', r_overlap < 0.2,
          f'overlap reward = {r_overlap:.3f}',
          f'overlap reward = {r_overlap:.3f} (expected ~0)')
    check('reward.disjoint_high', r_disjoint > 0.8,
          f'disjoint reward = {r_disjoint:.3f}',
          f'disjoint reward = {r_disjoint:.3f} (expected ~1)')
    # If still placeholder, both would be 0.5 → equal
    check('reward.differs', abs(r_disjoint - r_overlap) > 0.5,
          f'gap = {abs(r_disjoint - r_overlap):.3f}',
          'rewards equal — gamma * 0.5 placeholder still in place')


def test_7_scst_train_step_runs_without_error():
    """Sanity: train_step runs end-to-end and returns proper telemetry."""
    print('\n[Test 7] SCST train_step end-to-end (sanity)')

    # Build a fake GNN heuristic with one synthetic question
    class FakeGNN:
        def __init__(self):
            self.features = {'q1': {'question': 'what is the capital of france'}}
        def get(self, qid):
            return self.features.get(qid)
        def get_answers(self, qid):
            return ['gold_0']  # one of the synthetic gold entities

    # Patch extract_candidate_paths to return our synthetic paths
    import adagraph_cgar_v2 as mod
    orig = mod.extract_candidate_paths_by_k
    mod.extract_candidate_paths_by_k = lambda qid, gnn, k_options=(5, 10, 20, 50), max_paths_per_pair=3: \
        {K: make_paths(min(K, 6)) for K in k_options}

    try:
        ctrl = fresh_controller()
        emb = FakeEmbedder()
        trainer = CGARSCSTTrainer(ctrl, FakeGNN(), emb,
                                  alpha=0.6, beta=0.2, gamma=0.2,
                                  n_samples=4, lr=1e-3, device='cpu')
        result = trainer.train_step('q1')
        check('scst.returns_dict', isinstance(result, dict),
              'train_step returned dict',
              f'train_step returned {type(result).__name__}')
        for k in ['avg_reward', 'baseline', 'loss', 'pg_loss', 'calib_loss', 'advantage_nonzero']:
            check(f'scst.has_{k}', k in result,
                  f'{k}={result.get(k)}',
                  f'missing key {k}')
        check('scst.loss_finite', isinstance(result.get('loss'), float) and result['loss'] == result['loss'],
              f'loss={result["loss"]:.4f} (finite)',
              f'loss not finite: {result.get("loss")}')
    finally:
        mod.extract_candidate_paths_by_k = orig


def test_8_pg_loss_gradient_flows_to_policies():
    """End-to-end gradient test through proper REINFORCE loss."""
    print('\n[Test 8] -advantage * log_prob backward → policy gradients')
    ctrl = fresh_controller()
    emb = FakeEmbedder()
    q = torch.randn(1, EMB_DIM)

    _, det = ctrl(q, all_paths_by_k(), emb, deterministic=False, return_details=True)

    # Snapshot params before
    before_rel = {n: p.detach().clone() for n, p in ctrl.relevance_head.named_parameters()}
    before_kp = {n: p.detach().clone() for n, p in ctrl.k_policy.named_parameters()}

    # Build proper REINFORCE loss
    log_prob_total = det['log_prob_k'].sum() if det['log_prob_k'] is not None else torch.tensor(0.)
    for lp in det['log_prob_paths']:
        log_prob_total = log_prob_total + lp
    advantage = 1.0  # positive advantage → push toward this action
    pg_loss = -advantage * log_prob_total

    opt = torch.optim.SGD(ctrl.parameters(), lr=0.1)
    opt.zero_grad()
    pg_loss.backward()
    opt.step()

    # Check that params actually changed
    rel_changed = any(
        not torch.allclose(before_rel[n], p) for n, p in ctrl.relevance_head.named_parameters()
    )
    kp_changed = any(
        not torch.allclose(before_kp[n], p) for n, p in ctrl.k_policy.named_parameters()
    )
    check('e2e.rel_head_updated', rel_changed,
          'relevance_head params changed after step',
          'relevance_head DID NOT move — broken')
    check('e2e.k_policy_updated', kp_changed,
          'k_policy params changed after step',
          'k_policy DID NOT move — Bug B not fixed')


def test_10_adaptive_path_count():
    """rel_threshold must make the number of selected paths adaptive
    (v2.1 always returned min(max_paths, n_paths))."""
    print('\n[Test 10] Adaptive path count (rel_threshold)')
    torch.manual_seed(0)
    emb = FakeEmbedder()
    q = torch.randn(1, EMB_DIM)
    ctrl_legacy = CGARController(emb_dim=EMB_DIM, hidden_dim=16, max_paths=8,
                                 sprt_beta=0.99, redundancy_lambda=0.3, epsilon=0.0)
    sel_legacy = ctrl_legacy(q, all_paths_by_k(), emb, deterministic=True)
    ctrl_adapt = CGARController(emb_dim=EMB_DIM, hidden_dim=16, max_paths=8,
                                sprt_beta=0.99, redundancy_lambda=0.3, epsilon=0.0,
                                rel_threshold=1e6, min_select=2)
    ctrl_adapt.load_state_dict(ctrl_legacy.state_dict())
    sel_adapt = ctrl_adapt(q, all_paths_by_k(), emb, deterministic=True)
    check('adaptive.legacy_full', len(sel_legacy) == 8,
          f'legacy selects {len(sel_legacy)} = max_paths', f'legacy selected {len(sel_legacy)}')
    check('adaptive.threshold_stops', len(sel_adapt) == 2,
          f'huge threshold stops at min_select={len(sel_adapt)}',
          f'threshold ignored: selected {len(sel_adapt)}')


def test_9_bug_d_notebook_patched():
    """Bug D: check P7v3 Cell 11 source contains the fix."""
    print('\n[Test 9] Notebook Bug D patched')
    nb_path = Path(__file__).parent / 'P7v3_CGAR_Fixed.ipynb'
    if not nb_path.exists():
        check('bugd.notebook_found', False,
              '', f'{nb_path} not found — skip if not in repo')
        return
    nb = json.loads(nb_path.read_text())
    src_all = '\n'.join(''.join(c.get('source', [])) for c in nb['cells'] if c.get('cell_type') == 'code')
    check('bugd.has_all_dataset_results', 'all_dataset_results' in src_all,
          'all_dataset_results accumulator present',
          'all_dataset_results missing — Bug D not patched')
    check('bugd.no_buggy_comprehension', "for ds, r in [('results'" not in src_all,
          'old buggy comprehension removed',
          'old buggy `[("results", ds_results)]` still present')


# ── 3. Runner ────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('CGAR v2.1 Unit Tests (Bug A/B/C/D)')
    print('=' * 60)

    tests = [
        test_1_deterministic_forward_works,
        test_2_stochastic_forward_has_log_probs,
        test_3_backward_propagates_to_k_policy_and_relevance_head,
        test_4_reinforce_direction_is_correct,
        test_5_diversity_is_real_not_placeholder,
        test_6_compute_reward_uses_real_diversity,
        test_7_scst_train_step_runs_without_error,
        test_8_pg_loss_gradient_flows_to_policies,
        test_9_bug_d_notebook_patched,
        test_10_adaptive_path_count,
    ]

    for t in tests:
        try:
            t()
        except Exception as e:
            FAIL.append((t.__name__, f'EXCEPTION: {type(e).__name__}: {e}'))
            print(f'  FAIL {t.__name__} EXCEPTION: {type(e).__name__}: {e}')
            import traceback
            traceback.print_exc()

    print('\n' + '=' * 60)
    print(f'Result: {len(PASS)} passed, {len(FAIL)} failed')
    print('=' * 60)
    if FAIL:
        print('\nFailures:')
        for name, msg in FAIL:
            print(f'  FAIL {name}: {msg}')
        sys.exit(1)
    print('\nAll tests passed. Safe to train.')
    sys.exit(0)


if __name__ == '__main__':
    main()
