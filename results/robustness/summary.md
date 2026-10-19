# Robustness checks (clean patients)

## Part 1: stronger statistics, Test 1 (FFPE -> frozen)

AUROC = share of other candidates the true owner out-scores (0.5 = chance). p from permutations within the control's groups; Holm-corrected across all models x controls.

| Model | Candidates | Top-1 (x chance) | MRR (chance) | AUROC | p top-1 (Holm) | p MRR (Holm) |
| --- | --- | --- | --- | --- | --- | --- |
| conch | all (~188) | 3.7% (x7.0) | 0.109 (0.031) | 0.734 | 0.0075 | 0.0075 |
| conch | dx (~94) | 5.9% (x5.5) | 0.148 (0.055) | 0.703 | 0.0075 | 0.0075 |
| conch | dx+site (~9) | 36.0% (x2.0) | 0.559 (0.392) | 0.692 | 0.0075 | 0.0075 |
| conch | reverse: frozen -> FFPE | 4.8% (x9.0) | 0.106 (0.031) | 0.699 | 0.0005 (raw) | 0.0005 (raw) |
| musk | all (~188) | 2.7% (x5.0) | 0.086 (0.031) | 0.713 | 0.0075 | 0.0075 |
| musk | dx (~94) | 4.3% (x4.0) | 0.124 (0.055) | 0.689 | 0.0075 | 0.0075 |
| musk | dx+site (~9) | 32.6% (x1.8) | 0.517 (0.392) | 0.656 | 0.0075 | 0.0075 |
| musk | reverse: frozen -> FFPE | 3.7% (x7.0) | 0.102 (0.031) | 0.698 | 0.0005 (raw) | 0.0005 (raw) |
| pathgen | all (~188) | 4.3% (x8.0) | 0.092 (0.031) | 0.681 | 0.0075 | 0.0075 |
| pathgen | dx (~94) | 5.3% (x5.0) | 0.120 (0.055) | 0.663 | 0.0075 | 0.0075 |
| pathgen | dx+site (~9) | 29.2% (x1.6) | 0.486 (0.392) | 0.616 | 0.0075 | 0.0075 |
| pathgen | reverse: frozen -> FFPE | 3.2% (x6.0) | 0.083 (0.031) | 0.642 | 0.0010 (raw) | 0.0005 (raw) |
| quilt | all (~188) | 3.7% (x7.0) | 0.078 (0.031) | 0.659 | 0.0075 | 0.0075 |
| quilt | dx (~94) | 4.8% (x4.5) | 0.115 (0.055) | 0.652 | 0.0075 | 0.0075 |
| quilt | dx+site (~9) | 25.3% (x1.4) | 0.459 (0.392) | 0.594 | 0.0075 | 0.0075 |
| quilt | reverse: frozen -> FFPE | 2.1% (x4.0) | 0.067 (0.031) | 0.627 | 0.0135 (raw) | 0.0005 (raw) |
| plip | all (~188) | 3.7% (x7.0) | 0.086 (0.031) | 0.661 | 0.0075 | 0.0075 |
| plip | dx (~94) | 4.3% (x4.0) | 0.117 (0.055) | 0.648 | 0.0075 | 0.0075 |
| plip | dx+site (~9) | 25.3% (x1.4) | 0.464 (0.392) | 0.596 | 0.0075 | 0.0075 |
| plip | reverse: frozen -> FFPE | 3.2% (x6.0) | 0.072 (0.031) | 0.606 | 0.0010 (raw) | 0.0005 (raw) |

## Part 2: trained attacker (held-out patients, 4 folds x 10 repeats)

| Model | Method | Top-1 | Chance | x chance | MRR | AUROC |
| --- | --- | --- | --- | --- | --- | --- |
| conch | zero-shot | 11.3% | 2.1% | 5.3 | 0.254 | 0.735 |
| conch | centering | 20.4% | 2.1% | 9.6 | 0.351 | 0.789 |
| conch | CORAL | 17.1% | 2.1% | 8.0 | 0.329 | 0.785 |
| conch | Procrustes | 18.2% | 2.1% | 8.6 | 0.336 | 0.816 |
| conch | ridge | 14.9% | 2.1% | 7.0 | 0.311 | 0.812 |
| musk | zero-shot | 8.6% | 2.1% | 4.0 | 0.211 | 0.712 |
| musk | centering | 16.4% | 2.1% | 7.7 | 0.301 | 0.775 |
| musk | CORAL | 12.8% | 2.1% | 6.0 | 0.255 | 0.738 |
| musk | Procrustes | 18.1% | 2.1% | 8.5 | 0.328 | 0.814 |
| musk | ridge | 13.1% | 2.1% | 6.1 | 0.266 | 0.770 |
| pathgen | zero-shot | 8.4% | 2.1% | 3.9 | 0.195 | 0.682 |
| pathgen | centering | 13.1% | 2.1% | 6.2 | 0.269 | 0.734 |
| pathgen | CORAL | 12.2% | 2.1% | 5.7 | 0.244 | 0.716 |
| pathgen | Procrustes | 18.3% | 2.1% | 8.6 | 0.319 | 0.770 |
| pathgen | ridge | 11.7% | 2.1% | 5.5 | 0.242 | 0.727 |
| quilt | zero-shot | 6.8% | 2.1% | 3.2 | 0.178 | 0.659 |
| quilt | centering | 8.9% | 2.1% | 4.2 | 0.208 | 0.692 |
| quilt | CORAL | 9.6% | 2.1% | 4.5 | 0.210 | 0.687 |
| quilt | Procrustes | 16.9% | 2.1% | 7.9 | 0.311 | 0.777 |
| quilt | ridge | 12.6% | 2.1% | 5.9 | 0.260 | 0.745 |
| plip | zero-shot | 7.2% | 2.1% | 3.4 | 0.183 | 0.661 |
| plip | centering | 8.4% | 2.1% | 3.9 | 0.204 | 0.687 |
| plip | CORAL | 7.8% | 2.1% | 3.6 | 0.191 | 0.676 |
| plip | Procrustes | 12.8% | 2.1% | 6.0 | 0.266 | 0.758 |
| plip | ridge | 10.2% | 2.1% | 4.8 | 0.231 | 0.736 |

## Part 3: how the slide is summarized from its tiles

| Model | Pooling | Top-1 (x chance) | Top-10 | MRR | AUROC | Same hospital (x chance) |
| --- | --- | --- | --- | --- | --- | --- |
| conch | mean of 200 tiles (paper) | 3.7% (x7.0) | 26.1% | 0.109 | 0.734 | x2.00 |
| conch | mean of 25 tiles | 3.8% (x7.2) | 23.2% | 0.106 | 0.725 | x1.97 |
| conch | mean of 50 tiles | 4.4% (x8.2) | 23.3% | 0.110 | 0.729 | x2.01 |
| conch | mean of 100 tiles | 3.6% (x6.8) | 25.3% | 0.108 | 0.732 | x2.06 |
| conch | centered mean | 2.1% (x4.0) | 23.9% | 0.095 | 0.750 | x2.16 |
| conch | tile matching (Chamfer) | 3.7% (x7.0) | 33.0% | 0.137 | 0.772 | x2.44 |
| conch | tile voting | 4.3% (x8.0) | 27.1% | 0.117 | 0.760 | x2.31 |
| musk | mean of 200 tiles (paper) | 2.7% (x5.0) | 18.6% | 0.086 | 0.713 | x1.81 |
| musk | mean of 25 tiles | 2.9% (x5.4) | 19.1% | 0.088 | 0.707 | x1.73 |
| musk | mean of 50 tiles | 1.8% (x3.4) | 17.3% | 0.082 | 0.710 | x1.74 |
| musk | mean of 100 tiles | 2.3% (x4.4) | 17.8% | 0.086 | 0.713 | x1.69 |
| musk | centered mean | 1.1% (x2.0) | 22.3% | 0.087 | 0.738 | x2.12 |
| musk | tile matching (Chamfer) | 2.7% (x5.0) | 21.8% | 0.100 | 0.751 | x2.09 |
| musk | tile voting | 3.2% (x6.0) | 19.7% | 0.090 | 0.732 | x2.16 |
| pathgen | mean of 200 tiles (paper) | 4.3% (x8.0) | 17.6% | 0.092 | 0.681 | x1.62 |
| pathgen | mean of 25 tiles | 3.3% (x6.2) | 16.1% | 0.084 | 0.674 | x1.56 |
| pathgen | mean of 50 tiles | 3.5% (x6.6) | 16.4% | 0.086 | 0.678 | x1.62 |
| pathgen | mean of 100 tiles | 4.1% (x7.8) | 16.4% | 0.092 | 0.679 | x1.67 |
| pathgen | centered mean | 3.2% (x6.0) | 18.6% | 0.084 | 0.691 | x1.72 |
| pathgen | tile matching (Chamfer) | 4.3% (x8.0) | 20.7% | 0.103 | 0.714 | x1.62 |
| pathgen | tile voting | 5.3% (x10.0) | 19.7% | 0.110 | 0.721 | x1.59 |
| quilt | mean of 200 tiles (paper) | 3.7% (x7.0) | 14.4% | 0.078 | 0.659 | x1.41 |
| quilt | mean of 25 tiles | 2.6% (x4.8) | 14.4% | 0.073 | 0.653 | x1.35 |
| quilt | mean of 50 tiles | 3.2% (x6.0) | 13.3% | 0.076 | 0.659 | x1.49 |
| quilt | mean of 100 tiles | 3.3% (x6.2) | 13.0% | 0.077 | 0.659 | x1.41 |
| quilt | centered mean | 2.7% (x5.0) | 17.0% | 0.074 | 0.678 | x1.66 |
| quilt | tile matching (Chamfer) | 2.1% (x4.0) | 19.1% | 0.079 | 0.673 | x1.69 |
| quilt | tile voting | 2.1% (x4.0) | 17.0% | 0.071 | 0.673 | x1.66 |
| plip | mean of 200 tiles (paper) | 3.7% (x7.0) | 16.0% | 0.086 | 0.661 | x1.41 |
| plip | mean of 25 tiles | 3.0% (x5.6) | 15.6% | 0.077 | 0.656 | x1.32 |
| plip | mean of 50 tiles | 2.2% (x4.2) | 15.9% | 0.076 | 0.659 | x1.39 |
| plip | mean of 100 tiles | 3.4% (x6.4) | 15.4% | 0.082 | 0.661 | x1.39 |
| plip | centered mean | 2.7% (x5.0) | 14.4% | 0.075 | 0.671 | x1.47 |
| plip | tile matching (Chamfer) | 2.7% (x5.0) | 16.5% | 0.081 | 0.678 | x1.47 |
| plip | tile voting | 3.7% (x7.0) | 16.0% | 0.088 | 0.679 | x1.41 |