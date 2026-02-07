"""
Adamic-Adar (AA) + Dual correction (DC)

Scoring model:
  - AA > 0 edges:  score = AA * gate  (gate detects pseudo-cohesion via CN external ratio)
  - AA == 0 edges: score = anchor_scale * L3  (L3 rescue for pseudo-fragmentation)

Gate thresholds and anchor_scale are auto-calibrated on the validation set.
Node-level Local Clustering Coefficient (LCC) is used as an auxiliary trait
for progressive gating.

Usage:
  # Full model (gate + L3 rescue):
  python aa_dc.py --use_gate 1 --gate_mode progressive \\
      --auto_gate 1 --use_l3 1 --rescue_mode anchor --auto_beta 1 \\
      --name ogbl-collab

  # AA baseline (no gate, no L3):
  python aa_dc.py --use_gate 0 --use_l3 0 \\
      --name ogbl-collab
"""

import argparse
from typing import List, Set, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import scipy.sparse as ssp
import torch
from ogb.linkproppred import LinkPropPredDataset, Evaluator
from tqdm import tqdm


# ──────────────────────────────────────────────
# Graph utilities
# ──────────────────────────────────────────────
def build_adj(num_nodes: int, edges: np.ndarray) -> List[Set[int]]:
    """Build an unweighted adjacency list (set-based) from edge array."""
    adj: List[Set[int]] = [set() for _ in range(num_nodes)]
    for u, v in edges:
        u, v = int(u), int(v)
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)
    return adj


def build_weighted_adj(
    num_nodes: int, edges: np.ndarray, weights: np.ndarray
) -> ssp.csr_matrix:
    """Build a symmetric weighted adjacency matrix (CSR) from edges + weights."""
    row = edges[:, 0].astype(np.int64)
    col = edges[:, 1].astype(np.int64)
    row_full = np.concatenate([row, col])
    col_full = np.concatenate([col, row])
    w_full = np.concatenate([weights, weights]).astype(np.float32)
    A = ssp.csr_matrix((w_full, (row_full, col_full)), shape=(num_nodes, num_nodes))
    return A.tocsr()


def precompute_inv_log_deg(A: ssp.csr_matrix) -> np.ndarray:
    """Compute 1/log(degree) for AA weighting. Nodes with deg <= 1 get 0."""
    deg = np.array(A.sum(axis=1)).flatten().astype(np.float64)
    inv = np.zeros_like(deg, dtype=np.float64)
    mask = deg > 1.0
    inv[mask] = 1.0 / np.log(deg[mask])
    return inv.astype(np.float32)


def precompute_aa_matrix(A: ssp.csr_matrix, inv_log_deg: np.ndarray) -> ssp.csr_matrix:
    """Precompute A * diag(inv_log_deg) for vectorized AA scoring."""
    return A.multiply(inv_log_deg).tocsr()


# ──────────────────────────────────────────────
# Gate: CN External Ratio
# ──────────────────────────────────────────────
def cn_external_ratio(u: int, v: int, adj: List[Set[int]]) -> float:
    """Fraction of each common neighbor's links that point outside the local set.

    Local set = N(u) | N(v) | {u, v}.
    High ext_ratio signals pseudo-cohesion: common neighbors are externally
    oriented despite forming an apparent local cluster.
    """
    cn_set = adj[u] & adj[v]
    k = len(cn_set)
    if k == 0:
        return 0.0

    local_set = adj[u] | adj[v] | {u, v}
    total = 0.0
    for w in cn_set:
        adj_w = adj[w]
        deg_w = len(adj_w)
        if deg_w == 0:
            continue
        external = len(adj_w - local_set)
        total += external / deg_w
    return total / k


def compute_gate_value(
    ext_ratio: float,
    gate_mode: str,
    ext_threshold: float,
    ext_penalty: float,
    avg_lcc: float = 0.0,
    gate_thresholds: Optional[Dict[str, float]] = None,
) -> float:
    """Compute gate value from ext_ratio.

    gate_mode='progressive' (primary):
        Data-driven 2-stage gating with calibrated thresholds.
        Stage 1: ext_ratio > thresh_1 -> basic external orientation (gate=0.5)
        Stage 2: + avg_lcc > thresh_1 -> cross-fiber bridge pattern (gate=0.25)

    gate_mode='threshold' (used internally during calibration):
        Binary gate: ext_penalty if ext_ratio > ext_threshold, else 1.0
    """
    if gate_mode == "threshold":
        return ext_penalty if ext_ratio > ext_threshold else 1.0

    if gate_mode == "progressive":
        if gate_thresholds is None:
            return ext_penalty if ext_ratio > ext_threshold else 1.0

        gate = 1.0

        # Stage 1: Basic external orientation
        if ext_ratio > gate_thresholds.get("ext_thresh_1", ext_threshold):
            gate = 0.5

        # Stage 2: Clear cross-fiber bridge pattern
        if ext_ratio > gate_thresholds.get(
            "ext_thresh_2", ext_threshold + 0.1
        ) and avg_lcc > gate_thresholds.get("lcc_thresh_1", 0.5):
            gate *= 0.5  # -> 0.25

        return gate

    raise ValueError(f"Unknown gate_mode: {gate_mode}")


# ──────────────────────────────────────────────
# Weighted AA and L3
# ──────────────────────────────────────────────
def build_twohop_contrib_dict(u: int, A: ssp.csr_matrix) -> Dict[int, float]:
    """Build {node: weighted_contribution} for u's 2-hop neighborhood."""
    row_u = A.getrow(u)
    twohop = row_u.dot(A).tocsr()
    return {int(i): float(v) for i, v in zip(twohop.indices, twohop.data)}


def l3_score(
    v: int,
    A: ssp.csr_matrix,
    inv_log_deg: np.ndarray,
    twohop_dict: Dict[int, float],
) -> float:
    """Compute L3 score for edge (u, v) given u's precomputed 2-hop dict.

    L3 path: u -> N(u) -> 2hop(u) -> N(v) -> v.
    The score aggregates weighted contributions through 3-hop paths,
    penalized by 1/log(deg) at intermediate nodes.
    """
    row_v = A.getrow(v)
    nbrs = row_v.indices
    if nbrs.size == 0:
        return 0.0
    s = 0.0
    for t in nbrs:
        tv = twohop_dict.get(int(t), 0.0)
        if tv != 0.0:
            s += tv * float(inv_log_deg[int(t)])
    return float(s)


# ──────────────────────────────────────────────
# Neg-edge normalization (supports shared negatives)
# ──────────────────────────────────────────────
def normalize_neg_edges(pos_edge: np.ndarray, neg_edge_raw: np.ndarray):
    """Normalize negative edges into a standard format.

    Returns:
        (neg_edges, num_neg, mode) where mode is 'shared' or 'per_pos'.
    """
    num_pos = pos_edge.shape[0]

    # (num_pos, num_neg, 2): per-positive negatives
    if neg_edge_raw.ndim == 3 and neg_edge_raw.shape[-1] == 2:
        return neg_edge_raw.astype(np.int64), int(neg_edge_raw.shape[1]), "per_pos"

    if neg_edge_raw.ndim == 2:
        # (num_neg, 2): shared negatives (primary format for ogbl-collab)
        if neg_edge_raw.shape[1] == 2:
            total_neg = int(neg_edge_raw.shape[0])
            return neg_edge_raw.astype(np.int64), total_neg, "shared"

        # (num_pos, num_neg): destination-only format
        if neg_edge_raw.shape[0] == num_pos:
            num_neg = int(neg_edge_raw.shape[1])
            out = np.empty((num_pos, num_neg, 2), dtype=np.int64)
            out[:, :, 0] = pos_edge[:, 0:1]
            out[:, :, 1] = neg_edge_raw.astype(np.int64)
            return out, num_neg, "per_pos"

    raise ValueError(f"Unexpected neg_edge shape: {neg_edge_raw.shape}")


# ──────────────────────────────────────────────
# Node traits: Local Clustering Coefficient
# ──────────────────────────────────────────────
def compute_exact_lcc(A: ssp.csr_matrix) -> np.ndarray:
    """Compute exact LCC for all nodes via sparse matrix multiplication.

    LCC_i = triangles_i / (d_i * (d_i - 1) / 2)
    """
    A_bin = A.copy()
    A_bin.data = np.ones_like(A_bin.data, dtype=np.float32)
    deg = np.array(A_bin.sum(axis=1)).flatten()
    print("    Computing exact triangles via sparse matrix mult ...")

    triangles_2x = A_bin.dot(A_bin).multiply(A_bin).sum(axis=1).A1
    denom = deg * (deg - 1)
    lcc = np.zeros_like(deg, dtype=np.float32)
    mask = denom > 0
    lcc[mask] = (triangles_2x[mask] / denom[mask]).astype(np.float32)
    return lcc


def edge_lcc_bottleneck(
    edges: np.ndarray, lcc: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-edge LCC bottleneck (min) and openness (1 - min)."""
    u = edges[:, 0].astype(np.int64)
    v = edges[:, 1].astype(np.int64)
    bottleneck = np.minimum(lcc[u], lcc[v]).astype(np.float32)
    openness = (1.0 - bottleneck).astype(np.float32)
    return bottleneck, openness


# ──────────────────────────────────────────────
# Diagnostics helpers
# ──────────────────────────────────────────────
def quantile_stats(x: np.ndarray, qs=(0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)) -> str:
    """Format quantile summary as a single-line string."""
    if x.size == 0:
        return "(empty)"
    q = np.quantile(x.astype(np.float64), list(qs))
    return "mean={:.4f} std={:.4f} q={}".format(
        float(np.mean(x)), float(np.std(x)), np.array2string(q, precision=4)
    )


def rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Approximate Spearman rank correlation."""
    if x.size == 0 or y.size == 0:
        return float("nan")
    rx = x.argsort().argsort().astype(np.float64)
    ry = y.argsort().argsort().astype(np.float64)
    rx = (rx - rx.mean()) / (rx.std() + 1e-12)
    ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    return float(np.mean(rx * ry))


def summarize_aa_gate_traits(
    tag: str,
    edges: np.ndarray,
    aa: np.ndarray,
    gate: Optional[np.ndarray],
    ext_ratio: Optional[np.ndarray],
    lcc: Optional[np.ndarray],
    mask: np.ndarray,
) -> None:
    """Print summary statistics for edges matching the given mask."""
    idx = np.where(mask)[0]
    print(f"  [{tag}] n={idx.size}")
    if idx.size == 0:
        return

    aa_m = aa[idx]
    print(f"    AA:   {quantile_stats(aa_m)}")

    if gate is not None:
        g_m = gate[idx]
        print(f"    gate: {quantile_stats(g_m)}")
        print(f"    corr(rank): corr(AA,gate)={rank_corr(aa_m, g_m):.3f}")

    if ext_ratio is not None:
        ext_m = ext_ratio[idx]
        print(f"    ext_ratio: {quantile_stats(ext_m)}")

    if lcc is not None:
        bottleneck, openness = edge_lcc_bottleneck(edges[idx], lcc)
        print(f"    lcc_bottleneck: {quantile_stats(bottleneck)}")
        print(f"    lcc_openness:   {quantile_stats(openness)}")
        print(
            f"    corr(rank): corr(AA,lcc_bottleneck)={rank_corr(aa_m, bottleneck):.3f}"
        )
        if gate is not None:
            print(
                f"               corr(gate,lcc_bottleneck)={rank_corr(g_m, bottleneck):.3f}"
            )


def bin_summary_by_aa_quantiles(
    tag: str,
    aa: np.ndarray,
    gate: Optional[np.ndarray],
    lcc_bottleneck: Optional[np.ndarray],
    bins=(0.0, 0.5, 0.9, 0.99, 1.0),
) -> None:
    """Print conditional means by AA-quantile bins."""
    if aa.size == 0:
        print(f"  [{tag}] (empty)")
        return

    qs = np.quantile(aa.astype(np.float64), list(bins))
    qs = np.maximum.accumulate(qs)

    print(
        f"  [{tag}] AA-quantile bins: {bins} -> thresholds "
        f"{np.array2string(qs, precision=6)}"
    )
    for i in range(len(qs) - 1):
        lo, hi = qs[i], qs[i + 1]
        if i == len(qs) - 2:
            m = (aa >= lo) & (aa <= hi)
        else:
            m = (aa >= lo) & (aa < hi)
        n = int(m.sum())
        if n == 0:
            print(f"    bin{i}: [{lo:.6g}, {hi:.6g}) n=0")
            continue

        msg = f"    bin{i}: [{lo:.6g}, {hi:.6g}{']' if i == len(qs) - 2 else ')'} n={n}"
        msg += f" | AA_mean={float(aa[m].mean()):.6g}"
        if gate is not None:
            msg += f" gate_mean={float(gate[m].mean()):.6g}"
        if lcc_bottleneck is not None:
            msg += f" lcc_bottleneck_mean={float(lcc_bottleneck[m].mean()):.6g}"
        print(msg)


# ──────────────────────────────────────────────
# Edge scoring
# ──────────────────────────────────────────────
def score_edges(
    edges: np.ndarray,
    A: ssp.csr_matrix,
    A_invlog: ssp.csr_matrix,
    inv_log_deg: np.ndarray,
    adj: List[Set[int]],
    use_gate: bool,
    gate_mode: str,
    ext_threshold: float,
    ext_penalty: float,
    lcc: Optional[np.ndarray],
    use_l3: bool,
    rescue_mode: str,
    anchor_scale: float,
    gate_thresholds: Optional[Dict[str, float]] = None,
    collect_l3: bool = False,
    show_progress: bool = True,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
    Dict[str, object],
]:
    """Score all edges with AA (+ optional gate) and L3 rescue.

    Self-loops are forced to score 0 and excluded from all computations.

    Returns:
        scores:    (E,) final scores
        aa_vals:   (E,) raw AA values
        gate_vals: (E,) gate values (None if not use_gate)
        ext_vals:  (E,) ext_ratio values (None if not use_gate)
        stats:     diagnostic counters
    """
    E = edges.shape[0]
    scores = np.zeros(E, dtype=np.float32)
    aa_vals = np.zeros(E, dtype=np.float32)
    gate_vals = np.full(E, np.nan, dtype=np.float32)
    ext_vals = np.full(E, np.nan, dtype=np.float32)

    aa0_idx: List[int] = []
    self_loop_count = 0

    # Batch AA computation
    print("  Batch-computing AA scores ...")
    us = edges[:, 0]
    vs = edges[:, 1]
    rows_u = A[us]
    rows_v = A_invlog[vs]
    aa_batch = np.array(rows_u.multiply(rows_v).sum(axis=1)).flatten()

    it = range(E)
    if show_progress:
        it = tqdm(it, desc="AA(+gate)", leave=False)

    for i in it:
        u = int(edges[i, 0])
        v = int(edges[i, 1])

        if u == v:
            self_loop_count += 1
            aa0_idx.append(i)
            if use_gate:
                gate_vals[i] = 0.0
                ext_vals[i] = 0.0
            continue

        aa = float(aa_batch[i])
        aa_vals[i] = aa

        if aa == 0.0:
            aa0_idx.append(i)
            continue

        if use_gate:
            ext = cn_external_ratio(u, v, adj)
            ext_vals[i] = ext

            avg_lcc_val = 0.0
            if lcc is not None:
                avg_lcc_val = (float(lcc[u]) + float(lcc[v])) / 2.0

            g = compute_gate_value(
                ext,
                gate_mode,
                ext_threshold,
                ext_penalty,
                avg_lcc=avg_lcc_val,
                gate_thresholds=gate_thresholds,
            )
            gate_vals[i] = g
            scores[i] = float(aa * g)
        else:
            scores[i] = aa

    stats: Dict[str, object] = {
        "E": E,
        "aa0": len(aa0_idx),
        "aa_pos": E - len(aa0_idx),
        "self_loops": self_loop_count,
        "l3_called_u": 0,
        "l3_scored_edges": 0,
        "l3_nonzero": 0,
    }

    if collect_l3:
        stats["l3_values"] = np.full(E, np.nan, dtype=np.float32)

    # L3 rescue for AA==0 edges (anchor mode: score = anchor_scale * L3)
    if use_l3 and len(aa0_idx) > 0:
        idx_by_u: Dict[int, List[int]] = {}
        for i in aa0_idx:
            u = int(edges[i, 0])
            v = int(edges[i, 1])
            if u == v:
                continue
            idx_by_u.setdefault(u, []).append(i)

        u_list = list(idx_by_u.keys())
        if show_progress:
            u_list = tqdm(u_list, desc="L3(twohop by u)", leave=False)

        for u in u_list:
            twohop = build_twohop_contrib_dict(u, A)
            stats["l3_called_u"] += 1

            for i in idx_by_u[u]:
                v = int(edges[i, 1])
                l3 = l3_score(v, A, inv_log_deg, twohop)
                if collect_l3:
                    stats["l3_values"][i] = float(l3)
                if l3 != 0.0:
                    stats["l3_nonzero"] += 1
                stats["l3_scored_edges"] += 1

                # Place pseudo-fragmentation edges near the AA>0 boundary
                scores[i] = float(anchor_scale * l3)

    return (
        scores,
        aa_vals,
        gate_vals if use_gate else None,
        ext_vals if use_gate else None,
        stats,
    )


# ──────────────────────────────────────────────
# Gate threshold calibration (progressive mode)
# ──────────────────────────────────────────────
def calibrate_gate_thresholds(
    pos_ext_ratio: np.ndarray,
    pos_avg_lcc: np.ndarray,
    neg_ext_ratio: np.ndarray,
    neg_avg_lcc: np.ndarray,
    *,
    pos_protection_q: float = 0.95,
    margin: float = 0.10,
    neg_quantiles: tuple = (0.33, 0.67),
) -> Dict[str, float]:
    """Calibrate progressive gate thresholds from validation data.

    Principle:
      1. Protect positive distribution: use pos_protection_q quantile + margin
      2. Progressive negative targeting: use quantiles of problematic negatives
    """
    pos_ext_high = float(np.quantile(pos_ext_ratio, pos_protection_q))
    pos_lcc_high = float(np.quantile(pos_avg_lcc, pos_protection_q))
    ext_thresh_1 = pos_ext_high + margin
    lcc_thresh_1 = pos_lcc_high + margin

    neg_above_base = neg_ext_ratio > ext_thresh_1
    n_neg_above = int(neg_above_base.sum())

    if n_neg_above > 0:
        neg_ext_filtered = neg_ext_ratio[neg_above_base]
        ext_thresh_2 = float(np.quantile(neg_ext_filtered, neg_quantiles[0]))
    else:
        ext_thresh_2 = ext_thresh_1 + 0.1

    pos_violate_1 = int((pos_ext_ratio > ext_thresh_1).sum())
    pos_violate_2 = int(
        ((pos_ext_ratio > ext_thresh_2) & (pos_avg_lcc > lcc_thresh_1)).sum()
    )
    neg_hit_1 = int((neg_ext_ratio > ext_thresh_1).sum())
    neg_hit_2 = int(
        ((neg_ext_ratio > ext_thresh_2) & (neg_avg_lcc > lcc_thresh_1)).sum()
    )

    return {
        "ext_thresh_1": ext_thresh_1,
        "ext_thresh_2": ext_thresh_2,
        "lcc_thresh_1": lcc_thresh_1,
        "pos_ext_q": pos_ext_high,
        "pos_lcc_q": pos_lcc_high,
        "margin": margin,
        "pos_protection_q": pos_protection_q,
        "pos_violate_1": pos_violate_1,
        "pos_violate_2": pos_violate_2,
        "neg_hit_1": neg_hit_1,
        "neg_hit_2": neg_hit_2,
        "n_pos": len(pos_ext_ratio),
        "n_neg": len(neg_ext_ratio),
        "n_neg_above_base": n_neg_above,
    }


# ──────────────────────────────────────────────
# Anchor calibration and optimization
# ──────────────────────────────────────────────
def calibrate_anchor(
    kth_neg_at_K: float,
    l3_pos_aa0: np.ndarray,
    *,
    anchor_q: float = 0.90,
    l3_neg_aa0: Optional[np.ndarray] = None,
    neg_q: float = 0.99,
) -> Dict[str, float]:
    """Calibrate anchor_scale so that L3 pos edges land near the AA>0 boundary.

    L3 > 0 edges represent pseudo-fragmented connections that should belong
    to the AA > 0 world. anchor_scale places them near the Hits@K boundary:

        anchor_scale = kth@K / quantile(pos_L3, anchor_q)

    Safety is ensured by the rarity of neg L3 > 0 edges (~0.5%).
    """
    kth = float(kth_neg_at_K)
    Qpos = float(np.quantile(l3_pos_aa0, anchor_q)) if l3_pos_aa0.size > 0 else 1.0

    if Qpos <= 0.0 or kth <= 0.0 or not np.isfinite(Qpos) or not np.isfinite(kth):
        scale = 1e-6
    else:
        scale = kth / Qpos

    result: Dict[str, float] = {
        "anchor_scale": scale,
        "kth": kth,
        "Qpos": Qpos,
        "anchor_q": anchor_q,
        "expected_pos_score": float(scale * Qpos) if Qpos > 0 else 0.0,
    }

    if l3_neg_aa0 is not None and l3_neg_aa0.size > 0:
        Qneg = float(np.quantile(l3_neg_aa0, neg_q))
        expected_neg = float(scale * Qneg)
        result["Qneg"] = Qneg
        result["neg_q"] = neg_q
        result["expected_neg_score"] = expected_neg
        result["neg_kth_ratio"] = expected_neg / kth if kth > 0 else float("inf")
        result["neg_above_kth"] = int((l3_neg_aa0 * scale > kth).sum())
        result["n_neg_l3_nonzero"] = int(l3_neg_aa0.size)

    if l3_pos_aa0.size > 0:
        result["pos_above_kth"] = int((l3_pos_aa0 * scale > kth).sum())
        result["n_pos_l3_nonzero"] = int(l3_pos_aa0.size)

    return result


def optimize_anchor_q(
    y_pos_base: np.ndarray,
    y_neg_base: np.ndarray,
    l3_pos_vals: np.ndarray,
    l3_neg_vals: np.ndarray,
    kth_aa: float,
    K: int = 50,
    q_grid: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Optimize anchor_q by maximizing Hits@K on the validation set.

    Sweeps anchor_q over q_grid with gate_floor fixed at 0. For each q,
    computes anchor_scale = kth@K / quantile(pos_L3_nz, q) and simulates
    the resulting Hits@K without re-running score_edges.
    """
    if q_grid is None:
        q_grid = np.arange(0.50, 0.99, 0.02)

    pos_l3_mask = (~np.isnan(l3_pos_vals)) & (l3_pos_vals > 0)
    neg_l3_mask = (~np.isnan(l3_neg_vals)) & (l3_neg_vals > 0)

    l3_pos_nz = l3_pos_vals[pos_l3_mask]

    if l3_pos_nz.size == 0:
        return {
            "best_q": 0.9,
            "best_hits": 0.0,
            "best_anchor_scale": 1e-6,
            "grid": [],
            "note": "fallback - no L3 data",
        }

    best_q = 0.9
    best_hits = -1.0
    best_scale = 1e-6
    grid_results = []

    y_pos_work = y_pos_base.copy()
    y_neg_work = y_neg_base.copy()

    # Zero out AA==0 positions before sweep
    pos_aa0_l3zero = (~np.isnan(l3_pos_vals)) & (l3_pos_vals == 0)
    neg_aa0_l3zero = (~np.isnan(l3_neg_vals)) & (l3_neg_vals == 0)
    y_pos_work[pos_l3_mask] = 0.0
    y_pos_work[pos_aa0_l3zero] = 0.0
    y_neg_work[neg_l3_mask] = 0.0
    y_neg_work[neg_aa0_l3zero] = 0.0

    for q in q_grid:
        Qpos = float(np.quantile(l3_pos_nz, q))
        if Qpos <= 0:
            continue
        scale = kth_aa / Qpos

        y_pos_work[pos_l3_mask] = l3_pos_vals[pos_l3_mask] * scale
        y_neg_work[neg_l3_mask] = l3_neg_vals[neg_l3_mask] * scale

        if y_neg_work.size >= K:
            kth = float(np.partition(y_neg_work, -K)[-K])
        else:
            kth = float(np.max(y_neg_work))

        hits = float(np.mean(y_pos_work > kth))
        neg_intrusion = (
            int((y_neg_work[neg_l3_mask] > kth).sum()) if neg_l3_mask.any() else 0
        )
        pos_rescued = (
            int((y_pos_work[pos_l3_mask] > kth).sum()) if pos_l3_mask.any() else 0
        )

        grid_results.append(
            {
                "q": float(q),
                "scale": float(scale),
                "hits": float(hits),
                "kth": float(kth),
                "neg_intrusion": neg_intrusion,
                "pos_rescued": pos_rescued,
            }
        )

        if hits > best_hits:
            best_hits = hits
            best_q = float(q)
            best_scale = float(scale)

        # Reset for next iteration
        y_pos_work[pos_l3_mask] = 0.0
        y_neg_work[neg_l3_mask] = 0.0

    return {
        "best_q": best_q,
        "best_hits": best_hits,
        "best_anchor_scale": best_scale,
        "grid": grid_results,
    }


# ──────────────────────────────────────────────
# Shared-negative Hits@K
# ──────────────────────────────────────────────
def hits_at_k_shared(
    y_pos: np.ndarray, y_neg_shared: np.ndarray, K: int
) -> Tuple[float, Dict[str, float]]:
    """Compute Hits@K with shared negatives."""
    if y_neg_shared.size == 0:
        return 1.0, {
            "kth": float("-inf"),
            "pos_eq_kth_frac": 0.0,
            "neg_eq_kth_frac": 0.0,
        }

    if y_neg_shared.size < K:
        kth = float(np.max(y_neg_shared))
    else:
        kth = float(np.partition(y_neg_shared, -K)[-K])

    hits = float(np.mean(y_pos > kth))
    pos_eq = float(np.mean(y_pos == kth))
    neg_eq = float(np.mean(y_neg_shared == kth))
    return hits, {"kth": kth, "pos_eq_kth_frac": pos_eq, "neg_eq_kth_frac": neg_eq}


# ──────────────────────────────────────────────
# TopK diagnostics with traits
# ──────────────────────────────────────────────
def _topk_indices(scores: np.ndarray, K: int) -> np.ndarray:
    K = int(min(K, scores.size))
    if K <= 0:
        return np.array([], dtype=np.int64)
    idx = np.argpartition(scores, -K)[-K:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return idx.astype(np.int64)


def dump_topk_stats(
    title: str,
    edges: np.ndarray,
    scores: np.ndarray,
    aa: np.ndarray,
    gate: Optional[np.ndarray],
    ext_ratio: Optional[np.ndarray],
    lcc: Optional[np.ndarray],
    K: int,
    max_print: int,
    use_aa_mask_for_lcc: bool = True,
    split: str = "none",
) -> None:
    """Print top-K edge statistics and save to CSV."""
    M = int(scores.size)
    K_eff = min(int(K), M)
    if K_eff <= 0:
        return

    idx = _topk_indices(scores, K_eff)
    e = edges[idx]
    s = scores[idx]
    a_top = aa[idx]

    aa_pos_mask = a_top > 0.0
    aa_pos_frac = float(np.mean(aa_pos_mask))

    print(
        f"  [{title}] top{K_eff}: AA>0 frac={aa_pos_frac:.3f}  "
        f"AA==0 frac={1.0 - aa_pos_frac:.3f}"
    )
    print(
        f"             score: min={float(s.min()):.6g}  "
        f"median={float(np.median(s)):.6g}  max={float(s.max()):.6g}"
    )

    def _print_stat(label, data):
        if data is not None and data.size > 0:
            print(
                f"             {label}: min={float(np.nanmin(data)):.6g}  "
                f"median={float(np.nanmedian(data)):.6g}  "
                f"max={float(np.nanmax(data)):.6g}"
            )

    if gate is not None and np.any(aa_pos_mask):
        _print_stat("gate(AA>0)", gate[idx][aa_pos_mask])

    if ext_ratio is not None and np.any(aa_pos_mask):
        print(
            f"             ext_ratio(AA>0): "
            f"{quantile_stats(ext_ratio[idx][aa_pos_mask])}"
        )

    if lcc is not None:
        mask = aa_pos_mask if use_aa_mask_for_lcc else np.ones(K_eff, dtype=bool)
        suffix = "(AA>0)" if use_aa_mask_for_lcc else "(all)"

        if np.any(mask):
            e_masked = e[mask]
            u_l, v_l = lcc[e_masked[:, 0]], lcc[e_masked[:, 1]]
            bottleneck = np.minimum(u_l, v_l)
            openness = np.maximum(1.0 - u_l, 1.0 - v_l)
            print(f"             lcc_bottleneck{suffix}: {quantile_stats(bottleneck)}")
            print(f"             lcc_openness{suffix}:   {quantile_stats(openness)}")
            if use_aa_mask_for_lcc:
                print(f"             u_lcc(AA>0):           {quantile_stats(u_l)}")
                print(f"             v_lcc(AA>0):           {quantile_stats(v_l)}")

    # Save CSV
    P = min(int(max_print), K_eff, 100)
    rows = []
    for r in range(P):
        u, v = int(e[r, 0]), int(e[r, 1])
        sc, val_a = float(s[r]), float(a_top[r])
        val_g = float(gate[idx][r]) if gate is not None else float("nan")
        val_e = float(ext_ratio[idx][r]) if ext_ratio is not None else float("nan")

        ul = vl = bt = op = float("nan")
        if lcc is not None:
            ul, vl = float(lcc[u]), float(lcc[v])
            bt, op = min(ul, vl), max(1.0 - ul, 1.0 - vl)

        rows.append(
            {
                "rank": r,
                "u": u,
                "v": v,
                "score": sc,
                "AA": val_a,
                "gate": val_g,
                "ext_ratio": val_e,
                "u_lcc": ul,
                "v_lcc": vl,
                "lcc_bottleneck": bt,
                "lcc_openness": op,
            }
        )

    if rows:
        df = pd.DataFrame(rows)
        csv_path = f"{title}_top{P}_{split}.csv"
        df.to_csv(csv_path, index=False)
        print(f"             -> saved CSV: {csv_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="AA link prediction with pseudo-cohesion gate and L3 rescue."
    )
    parser.add_argument("--name", type=str, default="ogbl-collab")
    parser.add_argument(
        "--debug",
        type=int,
        default=1,
        help="1: rich diagnostics + official eval; 0: official eval only",
    )

    # Gate controls
    parser.add_argument("--use_gate", type=int, default=0)
    parser.add_argument(
        "--gate_mode",
        type=str,
        default="progressive",
        choices=["progressive", "threshold"],
        help=(
            "progressive: data-driven staged gating; "
            "threshold: binary ext_ratio gate (used internally during calibration)"
        ),
    )
    parser.add_argument(
        "--ext_threshold",
        type=float,
        default=0.5,
        help="ext_ratio threshold for pseudo-cohesion detection",
    )
    parser.add_argument(
        "--ext_penalty",
        type=float,
        default=0.5,
        help="Gate value when ext_ratio exceeds threshold (for threshold mode)",
    )
    parser.add_argument(
        "--auto_gate",
        type=int,
        default=0,
        help="If 1 and gate_mode=progressive: auto-calibrate thresholds on valid set",
    )

    # L3 rescue controls
    parser.add_argument("--use_l3", type=int, default=1)
    parser.add_argument(
        "--rescue_mode",
        type=str,
        default="anchor",
        choices=["anchor"],
        help="anchor: calibrate scale so L3 pos edges land near AA>0 boundary",
    )
    parser.add_argument(
        "--anchor_scale",
        type=float,
        default=1e-6,
        help="Scale factor for L3 rescue (auto-calibrated when auto_beta=1)",
    )
    parser.add_argument(
        "--auto_beta",
        type=int,
        default=0,
        help="If 1: auto-calibrate anchor_scale on valid set",
    )
    parser.add_argument(
        "--anchor_q",
        type=float,
        default=0.90,
        help="Pos L3 quantile mapped to kth@K boundary (auto-optimized when auto_beta=1)",
    )
    parser.add_argument(
        "--neg_q",
        type=float,
        default=0.99,
        help="Neg L3 quantile for safety validation",
    )

    # Evaluation controls
    parser.add_argument("--hits_ks", type=str, default="10,50,100")
    parser.add_argument("--dump_neg_topk", type=str, default="50")
    parser.add_argument("--dump_max_rows", type=int, default=100)

    # Trait controls
    parser.add_argument("--use_node_traits", type=int, default=1)

    # Extra diagnostics
    parser.add_argument("--print_bin_summary", type=int, default=1)
    parser.add_argument("--progress", type=int, default=1)

    args = parser.parse_args()

    debug = bool(int(args.debug))
    evaluator = Evaluator(name=args.name)

    use_gate = bool(args.use_gate)
    use_l3 = bool(args.use_l3)
    show_progress = debug and bool(args.progress)
    use_node_traits = bool(args.use_node_traits)
    print_bin_summary = debug and bool(args.print_bin_summary)

    Ks = [int(x) for x in args.hits_ks.split(",") if x.strip()]
    dumpKs = [int(x) for x in args.dump_neg_topk.split(",") if x.strip()]

    # ── Load dataset ──
    print(f"Loading dataset {args.name} ...")
    dataset = LinkPropPredDataset(name=args.name)
    split_edge = dataset.get_edge_split()
    graph = dataset[0]
    num_nodes = int(graph["num_nodes"])

    print("Loading edges and edge weights (with time decay) ...")
    train_edges = np.asarray(split_edge["train"]["edge"], dtype=np.int64)
    valid_edges = np.asarray(split_edge["valid"]["edge"], dtype=np.int64)

    # ── Time decay ──
    decay_rate = 0.95

    years_list = []
    if "edge_year" in graph:
        years_list.append(graph["edge_year"].flatten())
    if "year" in split_edge["train"]:
        years_list.append(np.array(split_edge["train"]["year"]).flatten())
    if "year" in split_edge["valid"]:
        years_list.append(np.array(split_edge["valid"]["year"]).flatten())

    if years_list:
        max_year = float(np.concatenate(years_list).max())
        print(f"  Time decay: max_year={int(max_year)}, decay_rate={decay_rate}")
    else:
        max_year = 2019.0
        print("  WARNING: No year info found. Using default max_year=2019.")

    # Train weights: base_weight (co-authorship count) * time_decay
    if "edge_weight" in graph and graph["edge_weight"] is not None:
        graph_edges = graph["edge_index"].T
        graph_weights_raw = np.asarray(graph["edge_weight"], dtype=np.float32).flatten()

        edge_to_weight: Dict[Tuple[int, int], float] = {}
        for i, (u, v) in enumerate(graph_edges):
            edge_to_weight[(int(u), int(v))] = float(graph_weights_raw[i])

        train_base_weights = np.array(
            [
                edge_to_weight.get(
                    (int(u), int(v)), edge_to_weight.get((int(v), int(u)), 1.0)
                )
                for u, v in train_edges
            ],
            dtype=np.float32,
        )
        print(
            f"  Train base weights: min={train_base_weights.min():.0f}, "
            f"max={train_base_weights.max():.0f}, mean={train_base_weights.mean():.2f}"
        )
    else:
        train_base_weights = np.ones(len(train_edges), dtype=np.float32)
        print("  No base edge_weight found, using 1.0.")

    if "year" in split_edge["train"]:
        train_years = np.asarray(
            split_edge["train"]["year"], dtype=np.float32
        ).flatten()
        train_time_weights = np.power(decay_rate, max_year - train_years)
        train_weights = train_base_weights * train_time_weights
        print(f"  -> Applied decay to train: final_mean={train_weights.mean():.4f}")
    else:
        train_weights = train_base_weights
        print("  WARNING: No train years found. Using base weights only.")

    # Valid weights: time decay only (no base weight provided)
    if "year" in split_edge["valid"]:
        valid_years = np.asarray(
            split_edge["valid"]["year"], dtype=np.float32
        ).flatten()
        valid_weights = np.power(decay_rate, max_year - valid_years)
        print(f"  -> Applied decay to valid: mean={valid_weights.mean():.4f}")
    else:
        valid_weights = np.ones(len(valid_edges), dtype=np.float32)
        print("  WARNING: No valid years found. Using 1.0 for valid weights.")

    # ── Build adjacency matrices ──
    print("Building weighted adjacency for TRAIN ...")
    A_train = build_weighted_adj(num_nodes, train_edges, train_weights)
    inv_log_deg_train = precompute_inv_log_deg(A_train)
    A_invlog_train = precompute_aa_matrix(A_train, inv_log_deg_train)
    adj_train = build_adj(num_nodes, train_edges)

    print("Building weighted adjacency for TEST (train + valid edges) ...")
    all_edges = np.vstack([train_edges, valid_edges])
    all_weights = np.concatenate([train_weights, valid_weights])
    A_test = build_weighted_adj(num_nodes, all_edges, all_weights)
    inv_log_deg_test = precompute_inv_log_deg(A_test)
    A_invlog_test = precompute_aa_matrix(A_test, inv_log_deg_test)
    adj_test = build_adj(num_nodes, all_edges)

    # ── Node traits (LCC) ──
    lcc_train = None
    lcc_test = None
    if use_node_traits:
        print("\nComputing node traits (exact LCC) via matrix ops ...")
        lcc_train = compute_exact_lcc(A_train)
        lcc_test = compute_exact_lcc(A_test)
        print("  Traits ready.")

    # ── Print scoring config ──
    print("\nScoring config:")
    print(f"  use_gate={use_gate} gate_mode={args.gate_mode}")
    print(f"  ext_threshold={args.ext_threshold} ext_penalty={args.ext_penalty}")
    print(
        f"  use_l3={use_l3} rescue_mode={args.rescue_mode} "
        f"anchor_scale={args.anchor_scale:g}"
    )
    if args.rescue_mode == "anchor":
        print(f"  anchor_q={args.anchor_q}")
    print("  Model: AA>0 -> AA * gate; AA==0 -> anchor_scale * L3")

    # Helper to build score_edges kwargs
    def _build_score_params(
        *,
        edges,
        A,
        A_invlog,
        inv_log_deg,
        adj,
        lcc,
        gate_thresholds_=None,
        use_l3_=use_l3,
        anchor_scale_=None,
        collect_l3_=False,
        gate_mode_=None,
    ):
        return dict(
            edges=edges,
            A=A,
            A_invlog=A_invlog,
            inv_log_deg=inv_log_deg,
            adj=adj,
            use_gate=use_gate,
            gate_mode=gate_mode_ if gate_mode_ is not None else args.gate_mode,
            ext_threshold=args.ext_threshold,
            ext_penalty=args.ext_penalty,
            lcc=lcc,
            use_l3=use_l3_,
            rescue_mode=args.rescue_mode,
            anchor_scale=(
                anchor_scale_ if anchor_scale_ is not None else args.anchor_scale
            ),
            gate_thresholds=gate_thresholds_,
            collect_l3=collect_l3_,
            show_progress=show_progress,
        )

    # ── Auto-calibrate gate thresholds (progressive mode) ──
    gate_thresholds = None
    if bool(args.auto_gate) and args.gate_mode == "progressive" and use_gate:
        print(
            "\n[auto_gate] Calibrating gate thresholds on valid set (train graph) ..."
        )

        pos_edge_v = np.asarray(split_edge["valid"]["edge"], dtype=np.int64)
        neg_edge_raw_v = np.asarray(split_edge["valid"]["edge_neg"])
        neg_edge_v, _, neg_mode_v = normalize_neg_edges(pos_edge_v, neg_edge_raw_v)

        if neg_mode_v != "shared":
            print(
                "  [auto_gate] WARNING: valid neg_mode is not shared; "
                "skipping calibration."
            )
        else:
            # Collect ext_ratio using threshold mode (binary gate for data collection)
            y_pos_v, aa_pos_v, gate_pos_v, ext_pos_v, stats_pos_v = score_edges(
                **_build_score_params(
                    edges=pos_edge_v,
                    A=A_train,
                    A_invlog=A_invlog_train,
                    inv_log_deg=inv_log_deg_train,
                    adj=adj_train,
                    lcc=lcc_train,
                    gate_mode_="threshold",
                    use_l3_=use_l3,
                    anchor_scale_=0.0,
                    collect_l3_=True,
                )
            )
            y_neg_v, aa_neg_v, gate_neg_v, ext_neg_v, stats_neg_v = score_edges(
                **_build_score_params(
                    edges=neg_edge_v,
                    A=A_train,
                    A_invlog=A_invlog_train,
                    inv_log_deg=inv_log_deg_train,
                    adj=adj_train,
                    lcc=lcc_train,
                    gate_mode_="threshold",
                    use_l3_=use_l3,
                    anchor_scale_=0.0,
                    collect_l3_=True,
                )
            )

            # Calibrate from top-K edges
            pos_aa_mask = aa_pos_v > 0.0
            neg_aa_mask = aa_neg_v > 0.0

            if pos_aa_mask.sum() > 0 and neg_aa_mask.sum() > 0:
                K_calib = 50
                pos_scores_aa = y_pos_v[pos_aa_mask]
                pos_aa_indices = np.where(pos_aa_mask)[0]
                pos_topk_idx = pos_aa_indices[np.argsort(pos_scores_aa)[-K_calib:]]

                neg_scores_aa = y_neg_v[neg_aa_mask]
                neg_aa_indices = np.where(neg_aa_mask)[0]
                if len(neg_aa_indices) >= K_calib:
                    neg_topk_idx = neg_aa_indices[np.argsort(neg_scores_aa)[-K_calib:]]
                else:
                    neg_topk_idx = neg_aa_indices

                pos_edges_topk = pos_edge_v[pos_topk_idx]
                neg_edges_topk = neg_edge_v[neg_topk_idx]
                pos_ext = ext_pos_v[pos_topk_idx]
                neg_ext = ext_neg_v[neg_topk_idx]

                pos_avg_lcc = np.array(
                    [(lcc_train[u] + lcc_train[v]) / 2.0 for u, v in pos_edges_topk],
                    dtype=np.float32,
                )
                neg_avg_lcc = np.array(
                    [(lcc_train[u] + lcc_train[v]) / 2.0 for u, v in neg_edges_topk],
                    dtype=np.float32,
                )

                gate_thresholds = calibrate_gate_thresholds(
                    pos_ext_ratio=pos_ext,
                    pos_avg_lcc=pos_avg_lcc,
                    neg_ext_ratio=neg_ext,
                    neg_avg_lcc=neg_avg_lcc,
                    pos_protection_q=0.95,
                    margin=0.10,
                    neg_quantiles=(0.33, 0.67),
                )

                print(f"  [auto_gate] Using top-{K_calib} edges for AA>0 calibration")
                print("  [auto_gate] Calibrated thresholds:")
                print(
                    f"    ext_thresh_1={gate_thresholds['ext_thresh_1']:.4f} "
                    f"(pos_q95 + margin)"
                )
                print(
                    f"    ext_thresh_2={gate_thresholds['ext_thresh_2']:.4f} "
                    f"(q33 of problematic neg)"
                )
                print(f"    lcc_thresh_1={gate_thresholds['lcc_thresh_1']:.4f}")
                print("  [auto_gate] Validation:")
                print(
                    f"    POS violations: "
                    f"S1={gate_thresholds['pos_violate_1']}/{gate_thresholds['n_pos']} "
                    f"S2={gate_thresholds['pos_violate_2']}/{gate_thresholds['n_pos']}"
                )
                print(
                    f"    NEG coverage:   "
                    f"S1={gate_thresholds['neg_hit_1']}/{gate_thresholds['n_neg']} "
                    f"S2={gate_thresholds['neg_hit_2']}/{gate_thresholds['n_neg']}"
                )
            else:
                print("  [auto_gate] WARNING: Not enough AA>0 edges for calibration.")

    # ── Auto-calibrate anchor_scale ──
    if bool(args.auto_beta) and use_l3:
        print("\n[auto_anchor] Calibrating anchor_scale on valid set (train graph) ...")

        pos_edge_v = np.asarray(split_edge["valid"]["edge"], dtype=np.int64)
        neg_edge_raw_v = np.asarray(split_edge["valid"]["edge_neg"])
        neg_edge_v, _, neg_mode_v = normalize_neg_edges(pos_edge_v, neg_edge_raw_v)

        if neg_mode_v != "shared":
            print("  [auto_anchor] WARNING: valid neg_mode is not shared; skipping.")
        else:
            # Score with anchor_scale=0 to get AA-only scores + collect L3 values
            y_pos_v, aa_pos_v, gate_pos_v, ext_pos_v, stats_pos_v = score_edges(
                **_build_score_params(
                    edges=pos_edge_v,
                    A=A_train,
                    A_invlog=A_invlog_train,
                    inv_log_deg=inv_log_deg_train,
                    adj=adj_train,
                    lcc=lcc_train,
                    gate_thresholds_=gate_thresholds,
                    anchor_scale_=0.0,
                    collect_l3_=True,
                )
            )
            y_neg_v, aa_neg_v, gate_neg_v, ext_neg_v, stats_neg_v = score_edges(
                **_build_score_params(
                    edges=neg_edge_v,
                    A=A_train,
                    A_invlog=A_invlog_train,
                    inv_log_deg=inv_log_deg_train,
                    adj=adj_train,
                    lcc=lcc_train,
                    gate_thresholds_=gate_thresholds,
                    anchor_scale_=0.0,
                    collect_l3_=True,
                )
            )

            # kth@50 from AA>0 negatives only
            neg_aa_mask_v = aa_neg_v > 0.0
            neg_scores_aa_v = y_neg_v[neg_aa_mask_v]
            K50 = 50
            if neg_scores_aa_v.size < K50:
                kth50_aa = (
                    float(np.max(neg_scores_aa_v)) if neg_scores_aa_v.size > 0 else 0.0
                )
            else:
                kth50_aa = float(np.partition(neg_scores_aa_v, -K50)[-K50])

            l3_pos_vals = stats_pos_v.get("l3_values", np.array([], dtype=np.float32))
            l3_neg_vals = stats_neg_v.get("l3_values", np.array([], dtype=np.float32))
            l3_pos_aa0 = l3_pos_vals[~np.isnan(l3_pos_vals)].astype(np.float32)
            l3_neg_aa0 = l3_neg_vals[~np.isnan(l3_neg_vals)].astype(np.float32)
            l3_pos_aa0_nz = l3_pos_aa0[l3_pos_aa0 > 0.0]
            l3_neg_aa0_nz = l3_neg_aa0[l3_neg_aa0 > 0.0]

            # Optimize anchor_q via grid search
            opt = optimize_anchor_q(
                y_pos_base=y_pos_v,
                y_neg_base=y_neg_v,
                l3_pos_vals=l3_pos_vals,
                l3_neg_vals=l3_neg_vals,
                kth_aa=kth50_aa,
                K=K50,
            )

            best_q = opt["best_q"]
            best_scale = opt["best_anchor_scale"]

            # Diagnostics via calibrate_anchor
            calib = calibrate_anchor(
                kth_neg_at_K=kth50_aa,
                l3_pos_aa0=l3_pos_aa0_nz,
                anchor_q=best_q,
                l3_neg_aa0=l3_neg_aa0_nz,
                neg_q=float(args.neg_q),
            )

            print(
                "  [auto_anchor] L3 nonzero (AA==0): pos_nz={}/{} neg_nz={}/{}".format(
                    l3_pos_aa0_nz.size,
                    l3_pos_aa0.size,
                    l3_neg_aa0_nz.size,
                    l3_neg_aa0.size,
                )
            )

            # Print grid results
            grid = opt["grid"]
            if grid:
                sorted_grid = sorted(grid, key=lambda x: -x["hits"])
                print(f"  [auto_anchor] Grid search: {len(grid)} q-points")
                print("  [auto_anchor] Top 5:")
                for rank, g in enumerate(sorted_grid[:5]):
                    marker = " <- BEST" if abs(g["q"] - best_q) < 1e-6 else ""
                    print(
                        f"    #{rank + 1} q={g['q']:.2f} scale={g['scale']:.6g} "
                        f"Hits@50={g['hits']:.6f} kth={g['kth']:.6g} "
                        f"pos_rescued={g['pos_rescued']} "
                        f"neg_intrusion={g['neg_intrusion']}{marker}"
                    )

            args.anchor_scale = float(best_scale)
            args.anchor_q = float(best_q)

            print(
                f"\n  [auto_anchor] OPTIMIZED: anchor_q={best_q:.4f} "
                f"-> anchor_scale={best_scale:.6g}"
            )
            print(f"  [auto_anchor] valid Hits@50={opt['best_hits']:.6f}")
            if "Qneg" in calib:
                print(
                    "  [auto_anchor] neg safety: Qneg({nq})={Qneg:.6g} "
                    "-> score={ns:.6g} (kth ratio={ratio:.4f})".format(
                        nq=calib.get("neg_q", "?"),
                        Qneg=calib["Qneg"],
                        ns=calib["expected_neg_score"],
                        ratio=calib["neg_kth_ratio"],
                    )
                )

    # ── Evaluate on valid and test splits ──
    for split in ["valid", "test"]:
        print(f"\nEvaluating split: {split}")

        if split == "valid":
            A, A_il, inv_ld, adj, lcc = (
                A_train,
                A_invlog_train,
                inv_log_deg_train,
                adj_train,
                lcc_train,
            )
        else:
            A, A_il, inv_ld, adj, lcc = (
                A_test,
                A_invlog_test,
                inv_log_deg_test,
                adj_test,
                lcc_test,
            )

        pos_edge = np.asarray(split_edge[split]["edge"], dtype=np.int64)
        neg_edge_raw = np.asarray(split_edge[split]["edge_neg"])
        neg_edge, num_neg, neg_mode = normalize_neg_edges(pos_edge, neg_edge_raw)
        if debug:
            print(
                f"  pos_edge: {pos_edge.shape}, neg_edge: {neg_edge_raw.shape}, "
                f"neg_mode={neg_mode}"
            )

        y_pos, aa_pos, gate_pos, ext_pos, stats_pos = score_edges(
            **_build_score_params(
                edges=pos_edge,
                A=A,
                A_invlog=A_il,
                inv_log_deg=inv_ld,
                adj=adj,
                lcc=lcc,
                gate_thresholds_=gate_thresholds,
            )
        )

        m_aa0_pos = aa_pos == 0.0
        n_pos = int(pos_edge.shape[0])
        n_pos_aa0 = int(m_aa0_pos.sum())
        if debug:
            print(
                f"  Pos AA==0: {n_pos_aa0} / {n_pos} "
                f"({n_pos_aa0 / n_pos * 100:.2f}%)"
            )

        if neg_mode != "shared":
            print(
                "  Official: (skipped) neg_mode is not shared; "
                "Hits@K expects shared negatives."
            )
            continue

        y_neg, aa_neg, gate_neg, ext_neg, stats_neg = score_edges(
            **_build_score_params(
                edges=neg_edge,
                A=A,
                A_invlog=A_il,
                inv_log_deg=inv_ld,
                adj=adj,
                lcc=lcc,
                gate_thresholds_=gate_thresholds,
            )
        )

        m_aa0_neg = aa_neg == 0.0
        if debug:
            print(
                f"  Neg(shared) AA==0: {int(m_aa0_neg.sum())} / {aa_neg.size} "
                f"({m_aa0_neg.mean() * 100:.2f}%)"
            )

        # OGB official evaluator
        input_dict = {
            "y_pred_pos": torch.from_numpy(y_pos).to(torch.float32),
            "y_pred_neg": torch.from_numpy(y_neg).to(torch.float32),
        }
        official = evaluator.eval(input_dict)
        official_str = "  Official: " + ", ".join(
            [f"{k}={float(v):.6g}" for k, v in official.items()]
        )
        print(official_str)

        if not debug:
            continue

        # TopK diagnostics
        for K in dumpKs:
            dump_topk_stats(
                title=f"pos top{K}",
                edges=pos_edge,
                scores=y_pos,
                aa=aa_pos,
                gate=gate_pos,
                ext_ratio=ext_pos,
                lcc=lcc,
                K=int(K),
                max_print=int(args.dump_max_rows),
                use_aa_mask_for_lcc=False,
                split=split,
            )
        for K in dumpKs:
            dump_topk_stats(
                title=f"neg top{K}",
                edges=neg_edge,
                scores=y_neg,
                aa=aa_neg,
                gate=gate_neg,
                ext_ratio=ext_neg,
                lcc=lcc,
                K=int(K),
                max_print=int(args.dump_max_rows),
                use_aa_mask_for_lcc=True,
                split=split,
            )

        # AA>0 global summaries
        print("  AA>0 global summaries:")
        summarize_aa_gate_traits(
            "pos AA>0",
            pos_edge,
            aa_pos,
            gate_pos,
            ext_pos,
            lcc,
            mask=(aa_pos > 0.0),
        )
        summarize_aa_gate_traits(
            "neg AA>0 (all shared)",
            neg_edge,
            aa_neg,
            gate_neg,
            ext_neg,
            lcc,
            mask=(aa_neg > 0.0),
        )

        # Conditional bins by AA quantile
        if print_bin_summary:
            print("  Conditional summaries by AA-quantile bins (AA>0 only):")
            pos_mask = aa_pos > 0.0
            neg_mask = aa_neg > 0.0

            pos_bt = (
                edge_lcc_bottleneck(pos_edge[pos_mask], lcc)[0]
                if lcc is not None
                else None
            )
            neg_bt = (
                edge_lcc_bottleneck(neg_edge[neg_mask], lcc)[0]
                if lcc is not None
                else None
            )

            bin_summary_by_aa_quantiles(
                "pos AA>0",
                aa=aa_pos[pos_mask],
                gate=(gate_pos[pos_mask] if gate_pos is not None else None),
                lcc_bottleneck=pos_bt,
            )
            bin_summary_by_aa_quantiles(
                "neg AA>0 (shared)",
                aa=aa_neg[neg_mask],
                gate=(gate_neg[neg_mask] if gate_neg is not None else None),
                lcc_bottleneck=neg_bt,
            )

        # Hits@K breakdown
        for K in Ks:
            hits, tm = hits_at_k_shared(y_pos, y_neg, K)
            kth = tm["kth"]
            print(
                f"  Hits@{K} (shared): {hits:.4f}  "
                f"(kth_neg={kth:.6g}, pos_eq_kth={tm['pos_eq_kth_frac']:.3f}, "
                f"neg_eq_kth={tm['neg_eq_kth_frac']:.3f})"
            )

            hits_aa0 = (
                float(np.mean(y_pos[m_aa0_pos] > kth))
                if n_pos_aa0 > 0
                else float("nan")
            )
            hits_aap = (
                float(np.mean(y_pos[~m_aa0_pos] > kth))
                if n_pos_aa0 < n_pos
                else float("nan")
            )
            print(
                f"           bucket: AA==0 Hits@{K}={hits_aa0:.4f}  |  "
                f"AA>0 Hits@{K}={hits_aap:.4f}"
            )

        if use_l3:
            print(
                f"  [pos L3 stats] aa0={stats_pos['aa0']} "
                f"aa_pos={stats_pos['aa_pos']} "
                f"l3_edges={stats_pos['l3_scored_edges']} "
                f"l3_nonzero={stats_pos['l3_nonzero']} "
                f"(twohop_u={stats_pos['l3_called_u']})"
            )
            print(
                f"  [neg L3 stats] aa0={stats_neg['aa0']} "
                f"aa_pos={stats_neg['aa_pos']} "
                f"l3_edges={stats_neg['l3_scored_edges']} "
                f"l3_nonzero={stats_neg['l3_nonzero']} "
                f"(twohop_u={stats_neg['l3_called_u']})"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
