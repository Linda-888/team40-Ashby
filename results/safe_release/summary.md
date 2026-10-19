# Safe ways to share: privacy vs usefulness

Found at #1 = attacker holding a patient's other slide picks the released item that contains them. Chance = 1 / number of released items. Accuracy = LUAD vs LUSC balanced accuracy on held-out patients (0.50 = coin flip).

## conch (188 patients, grouping: similar)

| Format | Found at #1 | Chance | x chance | Hidden among at least | Accuracy |
| --- | --- | --- | --- | --- | --- |
| raw (today) | 4.8% | 0.5% | 9.0x | 1 | 0.84 |
| groups of 5 | 6.9% | 2.8% | 2.5x | 5 | 0.81 |
| groups of 10 | 11.7% | 5.6% | 2.1x | 10 | 0.81 |
| groups of 20 | 20.7% | 12.5% | 1.7x | 20 | 0.81 |
| groups of 5 + DP eps=1 | 3.2% | 2.8% | 1.1x | 5 | 0.50 |
| groups of 5 + DP eps=4 | 3.7% | 2.8% | 1.3x | 5 | 0.54 |
| groups of 5 + DP eps=8 | 4.1% | 2.8% | 1.5x | 5 | 0.57 |
| groups of 10 + DP eps=1 | 6.2% | 5.6% | 1.1x | 10 | 0.52 |
| groups of 10 + DP eps=4 | 6.0% | 5.6% | 1.1x | 10 | 0.60 |
| groups of 10 + DP eps=8 | 7.6% | 5.6% | 1.4x | 10 | 0.68 |
| groups of 20 + DP eps=1 | 13.3% | 12.5% | 1.1x | 23 | 0.53 |
| groups of 20 + DP eps=4 | 15.2% | 12.5% | 1.2x | 23 | 0.64 |
| groups of 20 + DP eps=8 | 17.2% | 12.5% | 1.4x | 23 | 0.74 |
| predictions only | 0.7% | 0.5% | 1.3x | 1 | 0.84 |

## musk (188 patients, grouping: similar)

| Format | Found at #1 | Chance | x chance | Hidden among at least | Accuracy |
| --- | --- | --- | --- | --- | --- |
| raw (today) | 3.7% | 0.5% | 7.0x | 1 | 0.77 |
| groups of 5 | 8.0% | 2.8% | 2.9x | 5 | 0.74 |
| groups of 10 | 10.1% | 5.6% | 1.8x | 10 | 0.77 |
| groups of 20 | 21.8% | 12.5% | 1.7x | 20 | 0.79 |
| groups of 5 + DP eps=1 | 3.4% | 2.8% | 1.2x | 5 | 0.53 |
| groups of 5 + DP eps=4 | 3.2% | 2.8% | 1.1x | 5 | 0.54 |
| groups of 5 + DP eps=8 | 3.4% | 2.8% | 1.2x | 5 | 0.56 |
| groups of 10 + DP eps=1 | 6.6% | 5.6% | 1.2x | 10 | 0.49 |
| groups of 10 + DP eps=4 | 6.9% | 5.6% | 1.2x | 10 | 0.51 |
| groups of 10 + DP eps=8 | 7.4% | 5.6% | 1.3x | 10 | 0.52 |
| groups of 20 + DP eps=1 | 12.2% | 12.5% | 1.0x | 23 | 0.53 |
| groups of 20 + DP eps=4 | 14.9% | 12.5% | 1.2x | 23 | 0.55 |
| groups of 20 + DP eps=8 | 16.0% | 12.5% | 1.3x | 23 | 0.59 |
| predictions only | 0.9% | 0.5% | 1.7x | 1 | 0.77 |

## pathgen (188 patients, grouping: similar)

| Format | Found at #1 | Chance | x chance | Hidden among at least | Accuracy |
| --- | --- | --- | --- | --- | --- |
| raw (today) | 3.2% | 0.5% | 6.0x | 1 | 0.70 |
| groups of 5 | 6.9% | 2.8% | 2.5x | 5 | 0.69 |
| groups of 10 | 10.6% | 5.6% | 1.9x | 10 | 0.70 |
| groups of 20 | 13.8% | 12.5% | 1.1x | 20 | 0.67 |
| groups of 5 + DP eps=1 | 1.6% | 2.8% | 0.6x | 5 | 0.49 |
| groups of 5 + DP eps=4 | 1.6% | 2.8% | 0.6x | 5 | 0.50 |
| groups of 5 + DP eps=8 | 2.0% | 2.8% | 0.7x | 5 | 0.51 |
| groups of 10 + DP eps=1 | 5.7% | 5.6% | 1.0x | 10 | 0.51 |
| groups of 10 + DP eps=4 | 5.3% | 5.6% | 1.0x | 10 | 0.52 |
| groups of 10 + DP eps=8 | 5.9% | 5.6% | 1.1x | 10 | 0.51 |
| groups of 20 + DP eps=1 | 12.9% | 12.5% | 1.0x | 23 | 0.50 |
| groups of 20 + DP eps=4 | 13.8% | 12.5% | 1.1x | 23 | 0.53 |
| groups of 20 + DP eps=8 | 14.4% | 12.5% | 1.1x | 23 | 0.60 |
| predictions only | 1.2% | 0.5% | 2.3x | 1 | 0.70 |

## quilt (188 patients, grouping: similar)

| Format | Found at #1 | Chance | x chance | Hidden among at least | Accuracy |
| --- | --- | --- | --- | --- | --- |
| raw (today) | 2.1% | 0.5% | 4.0x | 1 | 0.68 |
| groups of 5 | 4.8% | 2.8% | 1.7x | 5 | 0.64 |
| groups of 10 | 10.1% | 5.6% | 1.8x | 10 | 0.66 |
| groups of 20 | 20.2% | 12.5% | 1.6x | 20 | 0.63 |
| groups of 5 + DP eps=1 | 3.5% | 2.8% | 1.3x | 5 | 0.51 |
| groups of 5 + DP eps=4 | 3.5% | 2.8% | 1.3x | 5 | 0.52 |
| groups of 5 + DP eps=8 | 3.4% | 2.8% | 1.2x | 5 | 0.53 |
| groups of 10 + DP eps=1 | 5.7% | 5.6% | 1.0x | 10 | 0.52 |
| groups of 10 + DP eps=4 | 6.7% | 5.6% | 1.2x | 10 | 0.52 |
| groups of 10 + DP eps=8 | 6.4% | 5.6% | 1.1x | 10 | 0.53 |
| groups of 20 + DP eps=1 | 12.2% | 12.5% | 1.0x | 23 | 0.50 |
| groups of 20 + DP eps=4 | 13.7% | 12.5% | 1.1x | 23 | 0.53 |
| groups of 20 + DP eps=8 | 13.3% | 12.5% | 1.1x | 23 | 0.55 |
| predictions only | 1.2% | 0.5% | 2.3x | 1 | 0.68 |

## plip (188 patients, grouping: similar)

| Format | Found at #1 | Chance | x chance | Hidden among at least | Accuracy |
| --- | --- | --- | --- | --- | --- |
| raw (today) | 3.2% | 0.5% | 6.0x | 1 | 0.72 |
| groups of 5 | 6.4% | 2.8% | 2.3x | 5 | 0.69 |
| groups of 10 | 5.9% | 5.6% | 1.1x | 10 | 0.68 |
| groups of 20 | 13.3% | 12.5% | 1.1x | 20 | 0.66 |
| groups of 5 + DP eps=1 | 4.1% | 2.8% | 1.5x | 5 | 0.47 |
| groups of 5 + DP eps=4 | 3.9% | 2.8% | 1.4x | 5 | 0.50 |
| groups of 5 + DP eps=8 | 3.5% | 2.8% | 1.3x | 5 | 0.52 |
| groups of 10 + DP eps=1 | 5.3% | 5.6% | 1.0x | 10 | 0.50 |
| groups of 10 + DP eps=4 | 5.7% | 5.6% | 1.0x | 10 | 0.51 |
| groups of 10 + DP eps=8 | 5.9% | 5.6% | 1.1x | 10 | 0.52 |
| groups of 20 + DP eps=1 | 12.1% | 12.5% | 1.0x | 23 | 0.50 |
| groups of 20 + DP eps=4 | 12.6% | 12.5% | 1.0x | 23 | 0.52 |
| groups of 20 + DP eps=8 | 12.4% | 12.5% | 1.0x | 23 | 0.54 |
| predictions only | 0.5% | 0.5% | 1.0x | 1 | 0.72 |

Notes: DP uses the classic Gaussian mechanism (delta = 1e-5); its formula is proven for eps <= 1 and is a standard approximation above that. Group membership itself is not hidden by DP here, only each average's value. A group hit reveals 'one of these k people', not the person.