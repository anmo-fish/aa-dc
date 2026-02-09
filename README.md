# AA-DC: Dual Correction for Adamic–Adar

**AA-DC** is a structural correction framework for link prediction.
It addresses two systematic biases that arise when relational feasibility is evaluated
under coarse recognition mappings, such as **Adamic–Adar (AA)**.

This repository contains the official implementation of: [Here](https://zenodo.org/records/18515023)

---

## Results

On **ogbl-collab**, AA-DC achieves competitive performance
*without any training loop* (GNN-free).

| Rank | Method | #Params | Test Hits@50 |
|:---|:---|:---|:---|
| 8 | Refined-GAE | 14M | 0.6816 ± 0.0041 |
| - | **AA + DC (Ours)** | **0** | **0.6802 ± 0.0000** |
| 9 | TopoLink | 483M | 0.6792 ± 0.0074 |
| 10 | S3GRL (PoS Plus) | 6M | 0.6683 ± 0.0030 |
| 11 | ELPH | 3M | 0.6636 ± 0.5876 |
| 12 | BUDDY | 1M | 0.6572 ± 0.0053 |
| 13 | AA + EPS | 0 | 0.6548 ± 0.0000 |
| 14 | SEAL-nofeat | 0.5M | 0.6474 ± 0.0043 |
| 15 | Adamic-Adar | 0 | 0.6417 ± 0.0000 |
| 16 | Common Neighbors | 0 | 0.6137 ± 0.0000 |

> **Note:** Leaderboard rankings and baseline results are current as of **February 6, 2026**.

---

## Requirements

**Latest tested combination:**
python == 3.12.11
numpy == 2.3.5
pandas == 3.0.0
scipy == 1.17.0
scikit-learn == 1.8.0
torch == 2.10.0+cu126
torch-geometric == 2.7.0
ogb == 1.3.6
networkx == 3.6.1
tqdm == 4.67.3

---

## Usage

The script `aa_dc.py` runs the full pipeline:
graph construction, metric computation, validation calibration,
and final test evaluation.

### Full Model (AA-DC)

```bash
python aa_dc.py   --use_gate 1   --gate_mode progressive   --auto_gate 1   --use_l3 1   --rescue_mode anchor   --auto_beta 1   --name ogbl-collab
```

### Ablation Studies

**Gate only (Correction I)**

```bash
python aa_dc.py   --use_gate 1   --use_l3 0   --gate_mode progressive   --auto_gate 1   --name ogbl-collab
```

**L3 only (Correction II)**

```bash
python aa_dc.py   --use_gate 0   --use_l3 1   --rescue_mode anchor   --auto_beta 1   --name ogbl-collab
```

**Adamic–Adar Baseline(time-decay)**

```bash
python aa_dc.py   --use_gate 0   --use_l3 0   --name ogbl-collab
```
**Notes**
- The baseline follows OGB leaderboard logic with time-decay weighting.
- All methods use the same weighted adjacency.
---

## Scoring Definition

### Case 1: `AA(u,v) > 0` (Base Signal)

```math
Score(u,v) = AA(u,v) \times Gate(ext\_ratio, LCC)
```

The gate suppresses pseudo-cohesive edges whose common neighbors
point away from the pair.

### Case 2: `AA(u,v) = 0` (Lost Signal)

```math
Score(u,v) = anchor\_scale \times L3(u,v)
```

- `L3(u,v)` counts length-3 paths `u → w → x → v`
- Paths are weighted by inverse log-degree
- `anchor_scale` is automatically calibrated so that rescued edges
  align with the `AA > 0` decision boundary

---
## Exmample(--debug 1)
```bash
Loading dataset ogbl-collab ...
Loading edges and edge weights (with time decay) ...
  Time decay: max_year=2018, decay_rate=0.95
  Train base weights: min=1, max=16, mean=1.11
  -> Applied decay to train: final_mean=0.7372
  -> Applied decay to valid: mean=1.0000
Building weighted adjacency for TRAIN ...
Building weighted adjacency for TEST (train + valid edges) ...

Computing node traits (exact LCC) via matrix ops ...
    Computing exact triangles via sparse matrix mult ...
    Computing exact triangles via sparse matrix mult ...
  Traits ready.

Scoring config:
  use_gate=True gate_mode=progressive
  ext_threshold=0.5 ext_penalty=0.5
  use_l3=True rescue_mode=anchor anchor_scale=1e-06
  anchor_q=0.9
  Model: AA>0 -> AA * gate; AA==0 -> anchor_scale * L3

[auto_gate] Calibrating gate thresholds on valid set (train graph) ...
  Batch-computing AA scores ...
  Batch-computing AA scores ...
  [auto_gate] Using top-50 edges for AA>0 calibration
  [auto_gate] Calibrated thresholds:
    ext_thresh_1=0.4734 (pos_q95 + margin)
    ext_thresh_2=0.7627 (q33 of problematic neg)
    lcc_thresh_1=0.5584
  [auto_gate] Validation:
    POS violations: S1=1/50 S2=0/50
    NEG coverage:   S1=33/50 S2=11/50

[auto_anchor] Calibrating anchor_scale on valid set (train graph) ...
  Batch-computing AA scores ...
  Batch-computing AA scores ...
  [auto_anchor] L3 nonzero (AA==0): pos_nz=8916/21101 neg_nz=553/99940
  [auto_anchor] Grid search: 25 q-points
  [auto_anchor] Top 5:
    #1 q=0.72 scale=0.0115628 Hits@50=0.673557 kth=0.0526124 pos_rescued=1933 neg_intrusion=8 <- BEST
    #2 q=0.60 scale=0.0205584 Hits@50=0.671976 kth=0.0747537 pos_rescued=2270 neg_intrusion=13
    #3 q=0.70 scale=0.0128099 Hits@50=0.671760 kth=0.0582869 pos_rescued=1933 neg_intrusion=8
    #4 q=0.74 scale=0.0104769 Hits@50=0.671743 kth=0.0520646 pos_rescued=1812 neg_intrusion=8
    #5 q=0.50 scale=0.0305891 Hits@50=0.671277 kth=0.0898526 pos_rescued=2647 neg_intrusion=18

  [auto_anchor] OPTIMIZED: anchor_q=0.7200 -> anchor_scale=0.0115628
  [auto_anchor] valid Hits@50=0.673557
  [auto_anchor] neg safety: Qneg(0.99)=5.7399 -> score=0.0663691 (kth ratio=1.7896)

Evaluating split: valid
  pos_edge: (60084, 2), neg_edge: (100000, 2), neg_mode=shared
  Batch-computing AA scores ...
  Pos AA==0: 21101 / 60084 (35.12%)
  Batch-computing AA scores ...
  Neg(shared) AA==0: 99941 / 100000 (99.94%)
  Official: hits@50=0.673557
  [pos top50] top50: AA>0 frac=1.000  AA==0 frac=0.000
             score: min=173.356  median=219.238  max=1457.19
             gate(AA>0): min=1  median=1  max=1
             ext_ratio(AA>0): mean=0.1512 std=0.1051 q=[0.0052 0.0313 0.0763 0.1276 0.2186 0.2909 0.4335]
             lcc_bottleneck(all): mean=0.1142 std=0.1133 q=[0.026  0.0383 0.0459 0.0784 0.1109 0.2316 0.4412]
             lcc_openness(all):   mean=0.8858 std=0.1133 q=[0.5588 0.7684 0.8891 0.9216 0.9541 0.9617 0.974 ]
             -> saved CSV: pos top50_top50_valid.csv
  [neg top50] top50: AA>0 frac=0.820  AA==0 frac=0.180
             score: min=0.0526124  median=0.148765  max=7.44962
             gate(AA>0): min=0.25  median=0.5  max=1
             ext_ratio(AA>0): mean=0.5465 std=0.2967 q=[0.     0.1176 0.2941 0.6021 0.802  0.9057 0.9889]
             lcc_bottleneck(AA>0): mean=0.3550 std=0.3101 q=[0.     0.0718 0.1218 0.2381 0.5273 1.     1.    ]
             lcc_openness(AA>0):   mean=0.6450 std=0.3101 q=[0.     0.     0.4727 0.7619 0.8782 0.9282 1.    ]
             u_lcc(AA>0):           mean=0.5778 std=0.3647 q=[0.0391 0.0942 0.2044 0.6429 1.     1.     1.    ]
             v_lcc(AA>0):           mean=0.5108 std=0.3416 q=[0.     0.1218 0.219  0.3929 0.9281 1.     1.    ]
             -> saved CSV: neg top50_top50_valid.csv
  AA>0 global summaries:
  [pos AA>0] n=38983
    AA:   mean=9.4811 std=23.5083 q=[2.6184e-02 4.2792e-01 1.2187e+00 2.9788e+00 8.0567e+00 2.2601e+01
 1.4572e+03]
    gate: mean=0.7914 std=0.2642 q=[0.25 0.5  0.5  1.   1.   1.   1.  ]
    corr(rank): corr(AA,gate)=0.395
    ext_ratio: mean=0.4093 std=0.2712 q=[0.     0.064  0.1795 0.3873 0.6209 0.812  0.9906]
    lcc_bottleneck: mean=0.2877 std=0.2683 q=[0.     0.0523 0.0857 0.1795 0.4168 0.7    1.    ]
    lcc_openness:   mean=0.7123 std=0.2683 q=[0.     0.3    0.5832 0.8205 0.9143 0.9477 1.    ]
    corr(rank): corr(AA,lcc_bottleneck)=-0.030
               corr(gate,lcc_bottleneck)=0.041
  [neg AA>0 (all shared)] n=59
    AA:   mean=0.6185 std=1.1985 q=[0.0139 0.0632 0.1352 0.2206 0.5521 1.2032 7.4496]
    gate: mean=0.5805 std=0.2854 q=[0.25  0.25  0.375 0.5   1.    1.    1.   ]
    corr(rank): corr(AA,gate)=0.416
    ext_ratio: mean=0.6388 std=0.2878 q=[0.     0.1692 0.4132 0.7576 0.8806 0.9167 0.9889]
    lcc_bottleneck: mean=0.3896 std=0.3337 q=[0.     0.069  0.1196 0.3048 0.564  1.     1.    ]
    lcc_openness:   mean=0.6104 std=0.3337 q=[0.     0.     0.436  0.6952 0.8804 0.931  1.    ]
    corr(rank): corr(AA,lcc_bottleneck)=-0.006
               corr(gate,lcc_bottleneck)=-0.131
  Conditional summaries by AA-quantile bins (AA>0 only):
  [pos AA>0] AA-quantile bins: (0.0, 0.5, 0.9, 0.99, 1.0) -> thresholds [2.618436e-02 2.978780e+00 2.260081e+01 1.018633e+02 1.457192e+03]
    bin0: [0.0261844, 2.97878) n=19491 | AA_mean=1.2816 gate_mean=0.691242 lcc_bottleneck_mean=0.297321
    bin1: [2.97878, 22.6008) n=15593 | AA_mean=8.01465 gate_mean=0.874287 lcc_bottleneck_mean=0.258613
    bin2: [22.6008, 101.863) n=3509 | AA_mean=45.9596 gate_mean=0.957823 lcc_bottleneck_mean=0.35581
    bin3: [101.863, 1457.19] n=390 | AA_mean=149.685 gate_mean=0.988462 lcc_bottleneck_mean=0.357413
  [neg AA>0 (shared)] AA-quantile bins: (0.0, 0.5, 0.9, 0.99, 1.0) -> thresholds [0.013933 0.220618 1.20323  5.398447 7.449624]
    bin0: [0.0139334, 0.220618) n=29 | AA_mean=0.118079 gate_mean=0.474138 lcc_bottleneck_mean=0.401815
    bin1: [0.220618, 1.20323) n=24 | AA_mean=0.469844 gate_mean=0.645833 lcc_bottleneck_mean=0.383491
    bin2: [1.20323, 5.39845) n=5 | AA_mean=2.86879 gate_mean=0.8 lcc_bottleneck_mean=0.317653
    bin3: [5.39845, 7.44962] n=1 | AA_mean=7.44962 gate_mean=1 lcc_bottleneck_mean=0.544615
  Hits@10 (shared): 0.5076  (kth_neg=0.595321, pos_eq_kth=0.000, neg_eq_kth=0.000)
           bucket: AA==0 Hits@10=0.0068  |  AA>0 Hits@10=0.7787
  Hits@50 (shared): 0.6736  (kth_neg=0.0526124, pos_eq_kth=0.000, neg_eq_kth=0.000)
           bucket: AA==0 Hits@50=0.0916  |  AA>0 Hits@50=0.9886
  Hits@100 (shared): 0.7158  (kth_neg=0.016804, pos_eq_kth=0.000, neg_eq_kth=0.000)
           bucket: AA==0 Hits@100=0.1914  |  AA>0 Hits@100=0.9997
  [pos L3 stats] aa0=21101 aa_pos=38983 l3_edges=21101 l3_nonzero=8916 (twohop_u=9038)
  [neg L3 stats] aa0=99941 aa_pos=59 l3_edges=99940 l3_nonzero=553 (twohop_u=76950)

Evaluating split: test
  pos_edge: (46329, 2), neg_edge: (100000, 2), neg_mode=shared
  Batch-computing AA scores ...
  Pos AA==0: 15936 / 46329 (34.40%)
  Batch-computing AA scores ...
  Neg(shared) AA==0: 99938 / 100000 (99.94%)
  Official: hits@50=0.680243
  [pos top50] top50: AA>0 frac=1.000  AA==0 frac=0.000
             score: min=213.575  median=238.727  max=1504.78
             gate(AA>0): min=1  median=1  max=1
             ext_ratio(AA>0): mean=0.1318 std=0.0986 q=[0.0042 0.0408 0.0833 0.0929 0.1591 0.2653 0.4491]
             lcc_bottleneck(all): mean=0.1952 std=0.1596 q=[0.0259 0.0465 0.0593 0.1169 0.3615 0.4293 0.4902]
             lcc_openness(all):   mean=0.8048 std=0.1596 q=[0.5098 0.5707 0.6385 0.8831 0.9407 0.9535 0.9741]
             -> saved CSV: pos top50_top50_test.csv
  [neg top50] top50: AA>0 frac=0.880  AA==0 frac=0.120
             score: min=0.0681364  median=0.179815  max=8.36797
             gate(AA>0): min=0.25  median=0.5  max=1
             ext_ratio(AA>0): mean=0.5955 std=0.2894 q=[0.0578 0.1583 0.3483 0.6933 0.8449 0.8925 0.9892]
             lcc_bottleneck(AA>0): mean=0.3439 std=0.2896 q=[0.     0.0748 0.1138 0.2539 0.4313 0.7941 1.    ]
             lcc_openness(AA>0):   mean=0.6561 std=0.2896 q=[0.     0.2059 0.5687 0.7461 0.8862 0.9252 1.    ]
             u_lcc(AA>0):           mean=0.5204 std=0.3555 q=[0.0381 0.0876 0.2026 0.3803 0.9713 1.     1.    ]
             v_lcc(AA>0):           mean=0.5153 std=0.3221 q=[0.     0.117  0.2627 0.4286 0.8083 1.     1.    ]
             -> saved CSV: neg top50_top50_test.csv
  AA>0 global summaries:
  [pos AA>0] n=30393
    AA:   mean=9.1402 std=25.4759 q=[2.8095e-02 4.4959e-01 1.2090e+00 3.1214e+00 8.0711e+00 1.9864e+01
 1.5048e+03]
    gate: mean=0.7485 std=0.2630 q=[0.25 0.5  0.5  1.   1.   1.   1.  ]
    corr(rank): corr(AA,gate)=0.384
    ext_ratio: mean=0.4659 std=0.2591 q=[0.     0.1091 0.2628 0.4615 0.6729 0.827  0.9928]
    lcc_bottleneck: mean=0.2304 std=0.2160 q=[0.     0.0497 0.079  0.1494 0.3111 0.5238 1.    ]
    lcc_openness:   mean=0.7696 std=0.2160 q=[0.     0.4762 0.6889 0.8506 0.921  0.9503 1.    ]
    corr(rank): corr(AA,lcc_bottleneck)=-0.064
               corr(gate,lcc_bottleneck)=0.035
  [neg AA>0 (all shared)] n=62
    AA:   mean=0.6274 std=1.2544 q=[0.0139 0.0672 0.145  0.2438 0.5268 1.1429 8.368 ]
    gate: mean=0.5887 std=0.2698 q=[0.25 0.25 0.5  0.5  1.   1.   1.  ]
    corr(rank): corr(AA,gate)=0.460
    ext_ratio: mean=0.6522 std=0.2846 q=[0.     0.1751 0.4444 0.7703 0.8798 0.9176 0.9892]
    lcc_bottleneck: mean=0.3729 std=0.3230 q=[0.     0.0713 0.115  0.276  0.4563 1.     1.    ]
    lcc_openness:   mean=0.6271 std=0.3230 q=[0.     0.     0.5437 0.724  0.885  0.9287 1.    ]
    corr(rank): corr(AA,lcc_bottleneck)=-0.016
               corr(gate,lcc_bottleneck)=-0.177
  Conditional summaries by AA-quantile bins (AA>0 only):
  [pos AA>0] AA-quantile bins: (0.0, 0.5, 0.9, 0.99, 1.0) -> thresholds [2.809546e-02 3.121363e+00 1.986437e+01 1.000104e+02 1.504776e+03]
    bin0: [0.0280955, 3.12136) n=15196 | AA_mean=1.31628 gate_mean=0.65634 lcc_bottleneck_mean=0.241976
    bin1: [3.12136, 19.8644) n=12157 | AA_mean=7.8752 gate_mean=0.817492 lcc_bottleneck_mean=0.21839
    bin2: [19.8644, 100.01) n=2736 | AA_mean=39.7962 gate_mean=0.927997 lcc_bottleneck_mean=0.216505
    bin3: [100.01, 1504.78] n=304 | AA_mean=174.915 gate_mean=0.981908 lcc_bottleneck_mean=0.258877
  [neg AA>0 (shared)] AA-quantile bins: (0.0, 0.5, 0.9, 0.99, 1.0) -> thresholds [0.013933 0.243763 1.142944 5.608464 8.367974]
    bin0: [0.0139334, 0.243763) n=31 | AA_mean=0.128615 gate_mean=0.516129 lcc_bottleneck_mean=0.409392
    bin1: [0.243763, 1.14294) n=24 | AA_mean=0.46688 gate_mean=0.604167 lcc_bottleneck_mean=0.321982
    bin2: [1.14294, 5.60846) n=6 | AA_mean=2.55633 gate_mean=0.833333 lcc_bottleneck_mean=0.387454
    bin3: [5.60846, 8.36797] n=1 | AA_mean=8.36797 gate_mean=1 lcc_bottleneck_mean=0.378378
  Hits@10 (shared): 0.5131  (kth_neg=0.595321, pos_eq_kth=0.000, neg_eq_kth=0.000)
           bucket: AA==0 Hits@10=0.0097  |  AA>0 Hits@10=0.7770
  Hits@50 (shared): 0.6802  (kth_neg=0.0681364, pos_eq_kth=0.000, neg_eq_kth=0.000)
           bucket: AA==0 Hits@50=0.0962  |  AA>0 Hits@50=0.9865
  Hits@100 (shared): 0.7220  (kth_neg=0.0237127, pos_eq_kth=0.000, neg_eq_kth=0.000)
           bucket: AA==0 Hits@100=0.1931  |  AA>0 Hits@100=0.9993 
  [pos L3 stats] aa0=15936 aa_pos=30393 l3_edges=15936 l3_nonzero=7257 (twohop_u=7793)
  [neg L3 stats] aa0=99938 aa_pos=62 l3_edges=99937 l3_nonzero=632 (twohop_u=76949)

Done.
```

---

## Citation

If you use this work, please cite:

```bibtex
@article{hirota2025aadc,
  author  = {Hirota, Daisuke},
  title   = {AA-DC: A Simple Dual Correction of Adamic-Adar for Pseudo-Cohesion and Pseudo-Fragmentation},
  year=2026,
  publisher= {Zenodo},
  doi={10.5281/zenodo.18515023},
  url={https://doi.org/10.5281/zenodo.18515023},
}

@article{hirota_2026_18411056,
  author = {Hirota, Daisuke},
  title={Space Reduction and Representability of Relational Feasibility : A Short Note Factorization under Coarsening Maps},
  year=2026,
  publisher= {Zenodo},
  doi={10.5281/zenodo.18411056},
  url={https://doi.org/10.5281/zenodo.18411056},
}

@article{hirota2026impossibilitycohesionfragmentation,
  title={The Impossibility of Cohesion Without Fragmentation}, 
  author={Daisuke Hirota},
  year={2026},
  eprint={2601.15317},
  archivePrefix={arXiv},
  primaryClass={physics.soc-ph},
  url={https://arxiv.org/abs/2601.15317}, 
}
```

---

## License

MIT License
