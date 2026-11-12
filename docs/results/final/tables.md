# Final results

Capture rate on the test stream: mean over training seeds ± half the seed range (n = seeds).

| Arm | iid-p0 | iid-p0.2 | iid-p0.5 | iid-p0.8 | iid-p1 |
|---|---|---|---|---|---|
| ctrl: TarMAC + position | 0.571 ± 0.001 (n=2) | 0.569 ± 0.009 (n=2) | 0.573 ± 0.008 (n=2) | 0.571 ± 0.000 (n=2) | 0.540 ± 0.025 (n=2) |
| A: point DIABL | 0.989 ± 0.017 (n=3) | 0.991 ± 0.014 (n=3) | 0.988 ± 0.017 (n=3) | 0.988 ± 0.019 (n=3) | 0.671 ± 0.170 (n=3) |
| D: map, true occupancy | 0.684 ± 0.058 (n=2) | 0.686 ± 0.062 (n=2) | 0.681 ± 0.065 (n=2) | 0.654 ± 0.070 (n=2) | 0.568 ± 0.037 (n=2) |
| C′: map, team posterior | 0.599 ± 0.008 (n=2) | 0.607 ± 0.007 (n=2) | 0.602 ± 0.005 (n=2) | 0.579 ± 0.022 (n=2) | 0.439 ± 0.103 (n=2) |
| F: A + stop-gradient | 1.000 ± 0.000 (n=3) | 1.000 ± 0.000 (n=3) | 1.000 ± 0.000 (n=3) | 1.000 ± 0.000 (n=3) | 0.694 ± 0.144 (n=3) |
| E: A without replay | 0.940 ± 0.089 (n=3) | 0.935 ± 0.097 (n=3) | 0.931 ± 0.103 (n=3) | 0.905 ± 0.142 (n=3) | 0.784 ± 0.203 (n=3) |
| AR: A, randomized-channel training | 0.987 ± 0.019 (n=3) | 0.988 ± 0.018 (n=3) | 0.989 ± 0.016 (n=3) | 1.000 ± 0.001 (n=3) | 0.999 ± 0.002 (n=3) |

## Hypotheses

| ID | Comparison | Seeds | Estimate | 95 % CI | Verdict |
|---|---|---|---|---|---|
| R1 | point_diabl@iid-p0.2 vs tarmac_pos@iid-p0.2 | [0, 1] | 1.734 | [1.701, 1.769] | confirmed |
| C1a | d_diabl_posterior@iid-p0 | [0, 1] | 0.041 | [0.039, 0.043] | not detected |
| C1b | d_diabl_posterior@iid-p0.2 vs d_diabl_coef10@iid-p0.2 | [0, 1] | -0.079 | [-0.091, -0.067] | not detected |
| FD | fd_heatmap@iid-p0 vs point_diabl@iid-p0 | [] | – | – | not run |
| FD-2 (secondary) | fd_heatmap@ge-p0.8-b5 vs point_diabl@ge-p0.8-b5 | [] | – | – | not run |
| M1 | point_diabl@iid-p0 vs point_diabl_stopgrad@iid-p0 | [0, 1, 2] | -0.011 | [-0.014, -0.009] | not detected |
| C2a | point_diabl@iid-p0-trigger-t3.5 vs point_diabl@iid-p0-random-r0.1 | [0, 1, 2] | – | – | not comparable |
| C2b (secondary) | point_diabl@iid-p0.8 vs point_diabl@ge-p0.8-b20 | [0, 1, 2] | 0.181 | [0.171, 0.191] | confirmed |
| M2 | point_diabl@iid-p0 vs point_diabl_noreplay@iid-p0 | [0, 1, 2] | 0.048 | [0.043, 0.054] | not detected |
| AR1 | point_diabl_robust@ge-p0.8-b20 vs point_diabl@ge-p0.8-b20 | [0, 1, 2] | 0.192 | [0.182, 0.202] | confirmed |
| AR2 | point_diabl_robust@iid-p0-random-r0.1 vs point_diabl@iid-p0-random-r0.1 | [0, 1, 2] | 0.069 | [0.059, 0.079] | confirmed |
| AR3 | point_diabl_robust@iid-p0 vs point_diabl@iid-p0 | [0, 1, 2] | -0.002 | [-0.007, 0.003] | not detected |
