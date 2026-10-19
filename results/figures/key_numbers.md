# Key numbers (re-identification as top-1, x chance)

## Across models (x chance)

- Test 1 slide -> slide: 10.0x to 12.0x  (CONCH 10.0x, MUSK 10.0x, PathGen-CLIP 12.0x, QuiltNet 12.0x, PLIP 10.0x)
- Test 1 same hospital: 1.4x to 2.1x  (CONCH 2.1x, MUSK 1.8x, PathGen-CLIP 1.7x, QuiltNet 1.5x, PLIP 1.4x)
- Test 2 report -> slide, zero-shot: 0.0x to 1.0x  (CONCH 0.0x, MUSK 1.0x, PathGen-CLIP 1.0x, QuiltNet 1.0x, PLIP 0.0x)
- Test 2 learned map: 3.9x to 5.9x  (CONCH 3.9x, MUSK 5.1x, PathGen-CLIP 5.8x, QuiltNet 4.6x, PLIP 5.9x)
- Test 2 learned map, unseen hospitals: 1.1x to 2.2x  (CONCH 1.5x, MUSK 1.5x, PathGen-CLIP 2.2x, QuiltNet 1.1x, PLIP 1.5x)
- Test 2 learned map, same hospital: 1.0x to 1.1x  (CONCH 1.1x, MUSK 1.1x, PathGen-CLIP 1.1x, QuiltNet 1.0x, PLIP 1.1x)
- Test 1 scrambled, 0 known pairs: 0.8x to 1.2x  (CONCH 0.8x, MUSK 1.2x, PathGen-CLIP 1.2x, QuiltNet 1.2x, PLIP 1.2x)
- Test 1 scrambled, 50 known pairs: 4.6x to 8.0x  (CONCH 7.6x, MUSK 8.0x, PathGen-CLIP 6.8x, QuiltNet 5.8x, PLIP 4.6x)
- Test 1 after noise: 1.7x to 2.0x  (CONCH 1.8x, MUSK 2.0x, PathGen-CLIP 1.8x, QuiltNet 1.7x, PLIP 1.8x)
- Test 1 after removing identity dirs: 5.5x to 7.0x  (CONCH 5.5x, MUSK 7.0x, PathGen-CLIP 7.0x, QuiltNet 7.0x, PLIP 6.0x)

## CONCH (193 patients)
- Test 1 slide -> slide, all: 5.2% vs 0.5% chance (10.0x, 95% CI 2.8%-9.3%, p=0.0005, n=193)
- Test 1 slide -> slide, same diagnosis: 6.7% vs 1.0% chance (6.5x, 95% CI 4.0%-11.2%, p=0.0005, n=193)
- Test 1 slide -> slide, same diagnosis + hospital: 36.8% vs 17.6% chance (2.1x, 95% CI 30.1%-44.0%, p=0.0005, n=182)
- Test 2 report -> slide, zero-shot, all: 0.0% vs 0.5% chance (0.0x, 95% CI 0.0%-2.0%, p=1.0000, n=193)
- Test 2 report -> slide, zero-shot, same diagnosis: 0.5% vs 1.0% chance (0.5x, 95% CI 0.1%-2.9%, p=0.8611, n=193)
- Test 2 report -> slide, zero-shot, same diagnosis + hospital: 19.2% vs 17.6% chance (1.1x, 95% CI 14.2%-25.6%, p=0.3118, n=182)
- Test 2 learned map: 3.9x (100 pairs)
- Test 2 learned map, unseen hospitals: 1.5x (100 pairs)
- Test 2 learned map, same hospital: 1.1x
- Test 1 scrambled, 0 known pairs: 0.8x
- Test 1 scrambled, 50 known pairs: 7.6x
- Test 1 after noise: 1.8x (sigma 3, acc 0.85->0.56)
- Test 1 after removing identity dirs: 5.5x (k=50, acc 0.79)

## MUSK (193 patients)
- Test 1 slide -> slide, all: 5.2% vs 0.5% chance (10.0x, 95% CI 2.8%-9.3%, p=0.0005, n=193)
- Test 1 slide -> slide, same diagnosis: 5.7% vs 1.0% chance (5.5x, 95% CI 3.2%-9.9%, p=0.0005, n=193)
- Test 1 slide -> slide, same diagnosis + hospital: 32.4% vs 17.6% chance (1.8x, 95% CI 26.0%-39.5%, p=0.0005, n=182)
- Test 2 report -> slide, zero-shot, all: 0.5% vs 0.5% chance (1.0x, 95% CI 0.1%-2.9%, p=0.6292, n=193)
- Test 2 report -> slide, zero-shot, same diagnosis: 0.5% vs 1.0% chance (0.5x, 95% CI 0.1%-2.9%, p=0.8611, n=193)
- Test 2 report -> slide, zero-shot, same diagnosis + hospital: 22.0% vs 17.6% chance (1.3x, 95% CI 16.6%-28.5%, p=0.0625, n=182)
- Test 2 learned map: 5.1x (100 pairs)
- Test 2 learned map, unseen hospitals: 1.5x (100 pairs)
- Test 2 learned map, same hospital: 1.1x
- Test 1 scrambled, 0 known pairs: 1.2x
- Test 1 scrambled, 50 known pairs: 8.0x
- Test 1 after noise: 2.0x (sigma 1.5, acc 0.83->0.57)
- Test 1 after removing identity dirs: 7.0x (k=50, acc 0.79)

## PathGen-CLIP (193 patients)
- Test 1 slide -> slide, all: 6.2% vs 0.5% chance (12.0x, 95% CI 3.6%-10.6%, p=0.0005, n=193)
- Test 1 slide -> slide, same diagnosis: 7.8% vs 1.0% chance (7.5x, 95% CI 4.8%-12.4%, p=0.0005, n=193)
- Test 1 slide -> slide, same diagnosis + hospital: 29.7% vs 17.6% chance (1.7x, 95% CI 23.5%-36.7%, p=0.0005, n=182)
- Test 2 report -> slide, zero-shot, all: 0.5% vs 0.5% chance (1.0x, 95% CI 0.1%-2.9%, p=0.6292, n=193)
- Test 2 report -> slide, zero-shot, same diagnosis: 1.0% vs 1.0% chance (1.0x, 95% CI 0.3%-3.7%, p=0.5917, n=193)
- Test 2 report -> slide, zero-shot, same diagnosis + hospital: 18.1% vs 17.6% chance (1.0x, 95% CI 13.2%-24.4%, p=0.4603, n=182)
- Test 2 learned map: 5.8x (100 pairs)
- Test 2 learned map, unseen hospitals: 2.2x (100 pairs)
- Test 2 learned map, same hospital: 1.1x
- Test 1 scrambled, 0 known pairs: 1.2x
- Test 1 scrambled, 50 known pairs: 6.8x
- Test 1 after noise: 1.8x (sigma 1.5, acc 0.78->0.55)
- Test 1 after removing identity dirs: 7.0x (k=50, acc 0.78)

## QuiltNet (193 patients)
- Test 1 slide -> slide, all: 6.2% vs 0.5% chance (12.0x, 95% CI 3.6%-10.6%, p=0.0005, n=193)
- Test 1 slide -> slide, same diagnosis: 6.7% vs 1.0% chance (6.5x, 95% CI 4.0%-11.2%, p=0.0005, n=193)
- Test 1 slide -> slide, same diagnosis + hospital: 25.8% vs 17.6% chance (1.5x, 95% CI 20.0%-32.6%, p=0.0045, n=182)
- Test 2 report -> slide, zero-shot, all: 0.5% vs 0.5% chance (1.0x, 95% CI 0.1%-2.9%, p=0.6292, n=193)
- Test 2 report -> slide, zero-shot, same diagnosis: 1.0% vs 1.0% chance (1.0x, 95% CI 0.3%-3.7%, p=0.5917, n=193)
- Test 2 report -> slide, zero-shot, same diagnosis + hospital: 16.5% vs 17.6% chance (0.9x, 95% CI 11.8%-22.6%, p=0.7071, n=182)
- Test 2 learned map: 4.6x (100 pairs)
- Test 2 learned map, unseen hospitals: 1.1x (100 pairs)
- Test 2 learned map, same hospital: 1.0x
- Test 1 scrambled, 0 known pairs: 1.2x
- Test 1 scrambled, 50 known pairs: 5.8x
- Test 1 after noise: 1.7x (sigma 1.5, acc 0.77->0.54)
- Test 1 after removing identity dirs: 7.0x (k=50, acc 0.73)

## PLIP (193 patients)
- Test 1 slide -> slide, all: 5.2% vs 0.5% chance (10.0x, 95% CI 2.8%-9.3%, p=0.0005, n=193)
- Test 1 slide -> slide, same diagnosis: 6.2% vs 1.0% chance (6.0x, 95% CI 3.6%-10.6%, p=0.0005, n=193)
- Test 1 slide -> slide, same diagnosis + hospital: 25.3% vs 17.6% chance (1.4x, 95% CI 19.5%-32.1%, p=0.0060, n=182)
- Test 2 report -> slide, zero-shot, all: 0.0% vs 0.5% chance (0.0x, 95% CI 0.0%-2.0%, p=1.0000, n=193)
- Test 2 report -> slide, zero-shot, same diagnosis: 0.0% vs 1.0% chance (0.0x, 95% CI 0.0%-2.0%, p=1.0000, n=193)
- Test 2 report -> slide, zero-shot, same diagnosis + hospital: 17.0% vs 17.6% chance (1.0x, 95% CI 12.3%-23.2%, p=0.6282, n=182)
- Test 2 learned map: 5.9x (100 pairs)
- Test 2 learned map, unseen hospitals: 1.5x (100 pairs)
- Test 2 learned map, same hospital: 1.1x
- Test 1 scrambled, 0 known pairs: 1.2x
- Test 1 scrambled, 50 known pairs: 4.6x
- Test 1 after noise: 1.8x (sigma 1.5, acc 0.82->0.55)
- Test 1 after removing identity dirs: 6.0x (k=50, acc 0.79)
