# Safe ways to share: privacy vs usefulness

Found at #1 = attacker holding a patient's other slide picks the released item that contains them. Chance = 1 / number of released items. Accuracy = LUAD vs LUSC balanced accuracy on held-out patients (0.50 = coin flip).

## musk (200 patients, grouping: similar)

| Format | Found at #1 | Chance | x chance | Hidden among at least | Accuracy |
| --- | --- | --- | --- | --- | --- |
| raw (today) | 8.0% | 0.5% | 16.0x | 1 | 0.77 |
| groups of 5 | 9.5% | 2.5% | 3.8x | 5 | 0.74 |
| groups of 10 | 9.0% | 5.0% | 1.8x | 10 | 0.74 |
| groups of 20 | 21.0% | 10.0% | 2.1x | 20 | 0.73 |
| groups of 5 + DP eps=1 | 2.0% | 2.5% | 0.8x | 5 | 0.50 |
| groups of 5 + DP eps=4 | 2.2% | 2.5% | 0.9x | 5 | 0.50 |
| groups of 5 + DP eps=8 | 2.2% | 2.5% | 0.9x | 5 | 0.52 |
| groups of 10 + DP eps=1 | 6.7% | 5.0% | 1.3x | 10 | 0.49 |
| groups of 10 + DP eps=4 | 7.5% | 5.0% | 1.5x | 10 | 0.51 |
| groups of 10 + DP eps=8 | 7.2% | 5.0% | 1.4x | 10 | 0.52 |
| groups of 20 + DP eps=1 | 9.7% | 10.0% | 1.0x | 20 | 0.49 |
| groups of 20 + DP eps=4 | 9.7% | 10.0% | 1.0x | 20 | 0.52 |
| groups of 20 + DP eps=8 | 10.3% | 10.0% | 1.0x | 20 | 0.54 |
| predictions only | 0.5% | 0.5% | 1.0x | 1 | 0.77 |

## pathgen (200 patients, grouping: similar)

| Format | Found at #1 | Chance | x chance | Hidden among at least | Accuracy |
| --- | --- | --- | --- | --- | --- |
| raw (today) | 5.5% | 0.5% | 11.0x | 1 | 0.72 |
| groups of 5 | 7.5% | 2.5% | 3.0x | 5 | 0.70 |
| groups of 10 | 9.5% | 5.0% | 1.9x | 10 | 0.70 |
| groups of 20 | 17.0% | 10.0% | 1.7x | 20 | 0.69 |
| groups of 5 + DP eps=1 | 2.2% | 2.5% | 0.9x | 5 | 0.47 |
| groups of 5 + DP eps=4 | 2.2% | 2.5% | 0.9x | 5 | 0.48 |
| groups of 5 + DP eps=8 | 2.3% | 2.5% | 0.9x | 5 | 0.49 |
| groups of 10 + DP eps=1 | 4.5% | 5.0% | 0.9x | 10 | 0.49 |
| groups of 10 + DP eps=4 | 5.2% | 5.0% | 1.0x | 10 | 0.51 |
| groups of 10 + DP eps=8 | 5.3% | 5.0% | 1.1x | 10 | 0.52 |
| groups of 20 + DP eps=1 | 12.5% | 10.0% | 1.2x | 20 | 0.50 |
| groups of 20 + DP eps=4 | 11.5% | 10.0% | 1.1x | 20 | 0.50 |
| groups of 20 + DP eps=8 | 12.3% | 10.0% | 1.2x | 20 | 0.54 |
| predictions only | 1.0% | 0.5% | 2.0x | 1 | 0.72 |

## quilt (200 patients, grouping: similar)

| Format | Found at #1 | Chance | x chance | Hidden among at least | Accuracy |
| --- | --- | --- | --- | --- | --- |
| raw (today) | 7.0% | 0.5% | 14.0x | 1 | 0.67 |
| groups of 5 | 8.5% | 2.5% | 3.4x | 5 | 0.66 |
| groups of 10 | 10.0% | 5.0% | 2.0x | 10 | 0.66 |
| groups of 20 | 17.0% | 10.0% | 1.7x | 20 | 0.65 |
| groups of 5 + DP eps=1 | 2.0% | 2.5% | 0.8x | 5 | 0.50 |
| groups of 5 + DP eps=4 | 2.0% | 2.5% | 0.8x | 5 | 0.50 |
| groups of 5 + DP eps=8 | 2.2% | 2.5% | 0.9x | 5 | 0.51 |
| groups of 10 + DP eps=1 | 4.7% | 5.0% | 0.9x | 10 | 0.48 |
| groups of 10 + DP eps=4 | 4.7% | 5.0% | 0.9x | 10 | 0.48 |
| groups of 10 + DP eps=8 | 5.2% | 5.0% | 1.0x | 10 | 0.48 |
| groups of 20 + DP eps=1 | 9.7% | 10.0% | 1.0x | 20 | 0.49 |
| groups of 20 + DP eps=4 | 10.0% | 10.0% | 1.0x | 20 | 0.51 |
| groups of 20 + DP eps=8 | 10.3% | 10.0% | 1.0x | 20 | 0.53 |
| predictions only | 1.2% | 0.5% | 2.3x | 1 | 0.67 |

## plip (200 patients, grouping: similar)

| Format | Found at #1 | Chance | x chance | Hidden among at least | Accuracy |
| --- | --- | --- | --- | --- | --- |
| raw (today) | 7.0% | 0.5% | 14.0x | 1 | 0.69 |
| groups of 5 | 7.0% | 2.5% | 2.8x | 5 | 0.67 |
| groups of 10 | 11.0% | 5.0% | 2.2x | 10 | 0.67 |
| groups of 20 | 13.0% | 10.0% | 1.3x | 20 | 0.67 |
| groups of 5 + DP eps=1 | 2.3% | 2.5% | 0.9x | 5 | 0.50 |
| groups of 5 + DP eps=4 | 2.3% | 2.5% | 0.9x | 5 | 0.50 |
| groups of 5 + DP eps=8 | 2.5% | 2.5% | 1.0x | 5 | 0.50 |
| groups of 10 + DP eps=1 | 5.2% | 5.0% | 1.0x | 10 | 0.50 |
| groups of 10 + DP eps=4 | 5.7% | 5.0% | 1.1x | 10 | 0.49 |
| groups of 10 + DP eps=8 | 6.3% | 5.0% | 1.3x | 10 | 0.51 |
| groups of 20 + DP eps=1 | 10.2% | 10.0% | 1.0x | 20 | 0.53 |
| groups of 20 + DP eps=4 | 10.3% | 10.0% | 1.0x | 20 | 0.54 |
| groups of 20 + DP eps=8 | 10.3% | 10.0% | 1.0x | 20 | 0.56 |
| predictions only | 1.2% | 0.5% | 2.3x | 1 | 0.69 |

Notes: DP uses the classic Gaussian mechanism (delta = 1e-5); its formula is proven for eps <= 1 and is a standard approximation above that. Group membership itself is not hidden by DP here, only each average's value. A group hit reveals 'one of these k people', not the person.