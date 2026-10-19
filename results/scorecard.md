# Privacy scorecard

|  | conch | musk | pathgen | quilt | plip |
|---|---|---|---|---|---|
| slide-to-slide (all) | 7.0% vs 0.5% chance (n=200) | 7.5% vs 0.5% chance (n=200) | 6.5% vs 0.5% chance (n=200) | 8.5% vs 0.5% chance (n=200) | 7.0% vs 0.5% chance (n=200) |
| slide-to-slide (same dx + site) | 38.3% vs 17.0% chance (n=188) | 34.6% vs 17.0% chance (n=188) | 30.9% vs 17.0% chance (n=188) | 28.2% vs 17.0% chance (n=188) | 27.7% vs 17.0% chance (n=188) |
| report-to-slide (all) | 0.0% vs 0.5% chance (n=200) | 0.5% vs 0.5% chance (n=200) | 0.5% vs 0.5% chance (n=200) | 0.5% vs 0.5% chance (n=200) | 0.0% vs 0.5% chance (n=200) |
| report-to-slide (same dx + site) HEADLINE | 18.6% vs 17.0% chance (n=188) | 22.3% vs 17.0% chance (n=188) | 17.6% vs 17.0% chance (n=188) | 16.0% vs 17.0% chance (n=188) | 17.0% vs 17.0% chance (n=188) |
| key scrambling, test1 | broken with 50 known pairs (procrustes) | broken with 50 known pairs (procrustes) | broken with 30 known pairs (procrustes) | broken with 75 known pairs (procrustes) | broken with 75 known pairs (procrustes) |
| key scrambling, test2 | n/a: no clear-text signal (see learned linkage) | n/a: no clear-text signal (see learned linkage) | n/a: no clear-text signal (see learned linkage) | n/a: no clear-text signal (see learned linkage) | n/a: no clear-text signal (see learned linkage) |
| learned linkage, report-to-slide | 5.2% vs 1.2% chance (x4.2) with 120 training pairs (procrustes); shuffled-pairs control x0.6 | 7.9% vs 1.2% chance (x6.3) with 120 training pairs (procrustes); shuffled-pairs control x0.6 | 9.1% vs 1.2% chance (x7.3) with 120 training pairs (procrustes); shuffled-pairs control x1.4 | 6.6% vs 1.2% chance (x5.3) with 120 training pairs (procrustes); shuffled-pairs control x1.0 | 7.9% vs 1.2% chance (x6.3) with 120 training pairs (procrustes); shuffled-pairs control x1.4 |
| learned linkage, report-to-slide (unseen hospitals) | 1.4% vs 1.1% chance (x1.3) with 100 training pairs (procrustes); shuffled-pairs control x1.0 | 0.9% vs 1.1% chance (x0.8) with 100 training pairs (procrustes); shuffled-pairs control x1.0 | 2.7% vs 1.1% chance (x2.4) with 100 training pairs (procrustes); shuffled-pairs control x1.1 | 1.2% vs 1.1% chance (x1.0) with 100 training pairs (procrustes); shuffled-pairs control x1.0 | 2.0% vs 1.1% chance (x1.8) with 100 training pairs (procrustes); shuffled-pairs control x2.0 |
| fix: noise | no setting blocks both tests | strength 2, accuracy 0.780 -> 0.560 | strength 2, accuracy 0.775 -> 0.508 | strength 1.5, accuracy 0.785 -> 0.498 | strength 1.5, accuracy 0.770 -> 0.498 |
| fix: remove | no setting blocks both tests | no setting blocks both tests | no setting blocks both tests | no setting blocks both tests | no setting blocks both tests |
