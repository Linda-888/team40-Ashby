# Key numbers (re-identification as top-1, x chance)

## Across models (x chance)

- Test 1 slide -> slide: 13.0x to 17.0x  (CONCH 14.0x, MUSK 15.0x, PathGen-CLIP 13.0x, QuiltNet 17.0x, PLIP 14.0x)
- Test 1 same hospital: 1.6x to 2.2x  (CONCH 2.2x, MUSK 2.0x, PathGen-CLIP 1.8x, QuiltNet 1.7x, PLIP 1.6x)
- Test 2 report -> slide, zero-shot: 0.0x to 1.0x  (CONCH 0.0x, MUSK 1.0x, PathGen-CLIP 1.0x, QuiltNet 1.0x, PLIP 0.0x)
- Test 2 learned map: 4.2x to 7.3x  (CONCH 4.2x, MUSK 6.3x, PathGen-CLIP 7.3x, QuiltNet 5.3x, PLIP 6.3x)
- Test 2 learned map, unseen hospitals: 0.8x to 2.4x  (CONCH 1.3x, MUSK 0.8x, PathGen-CLIP 2.4x, QuiltNet 1.0x, PLIP 1.8x)
- Test 2 learned map, same hospital: 1.0x to 1.1x  (CONCH 1.1x, MUSK 1.1x, PathGen-CLIP 1.1x, QuiltNet 1.0x, PLIP 1.0x)
- Test 1 scrambled, 0 known pairs: 0.8x to 1.4x  (CONCH 0.8x, MUSK 1.4x, PathGen-CLIP 1.2x, QuiltNet 1.2x, PLIP 1.2x)
- Test 1 scrambled, 50 known pairs: 5.6x to 8.8x  (CONCH 8.8x, MUSK 7.6x, PathGen-CLIP 7.2x, QuiltNet 5.6x, PLIP 5.8x)
- Test 1 after noise: 1.3x to 1.8x  (CONCH 1.8x, MUSK 1.3x, PathGen-CLIP 1.5x, QuiltNet 1.7x, PLIP 1.8x)
- Test 1 after removing identity dirs: 8.5x to 13.0x  (CONCH 13.0x, MUSK 8.5x, PathGen-CLIP 9.5x, QuiltNet 8.5x, PLIP 8.5x)

## CONCH (200 patients)
- Test 1 slide -> slide, all: 7.0% vs 0.5% chance (14.0x, 95% CI 4.2%-11.4%, p=0.0005, n=200)
- Test 1 slide -> slide, same diagnosis: 9.0% vs 1.0% chance (9.0x, 95% CI 5.8%-13.8%, p=0.0005, n=200)
- Test 1 slide -> slide, same diagnosis + hospital: 38.3% vs 17.0% chance (2.2x, 95% CI 31.6%-45.4%, p=0.0005, n=188)
- Test 2 report -> slide, zero-shot, all: 0.0% vs 0.5% chance (0.0x, 95% CI 0.0%-1.9%, p=1.0000, n=200)
- Test 2 report -> slide, zero-shot, same diagnosis: 0.5% vs 1.0% chance (0.5x, 95% CI 0.1%-2.8%, p=0.8611, n=200)
- Test 2 report -> slide, zero-shot, same diagnosis + hospital: 18.6% vs 17.0% chance (1.1x, 95% CI 13.7%-24.8%, p=0.3053, n=188)
- Test 2 learned map: 4.2x (120 pairs)
- Test 2 learned map, unseen hospitals: 1.3x (100 pairs)
- Test 2 learned map, same hospital: 1.1x
- Test 1 scrambled, 0 known pairs: 0.8x
- Test 1 scrambled, 50 known pairs: 8.8x
- Test 1 after noise: 1.8x (sigma 5, acc 0.83->0.50)
- Test 1 after removing identity dirs: 13.0x (k=100, acc 0.81)

## MUSK (200 patients)
- Test 1 slide -> slide, all: 7.5% vs 0.5% chance (15.0x, 95% CI 4.6%-12.0%, p=0.0005, n=200)
- Test 1 slide -> slide, same diagnosis: 8.0% vs 1.0% chance (8.0x, 95% CI 5.0%-12.6%, p=0.0005, n=200)
- Test 1 slide -> slide, same diagnosis + hospital: 34.6% vs 17.0% chance (2.0x, 95% CI 28.1%-41.6%, p=0.0005, n=188)
- Test 2 report -> slide, zero-shot, all: 0.5% vs 0.5% chance (1.0x, 95% CI 0.1%-2.8%, p=0.6307, n=200)
- Test 2 report -> slide, zero-shot, same diagnosis: 0.5% vs 1.0% chance (0.5x, 95% CI 0.1%-2.8%, p=0.8611, n=200)
- Test 2 report -> slide, zero-shot, same diagnosis + hospital: 22.3% vs 17.0% chance (1.3x, 95% CI 17.0%-28.8%, p=0.0305, n=188)
- Test 2 learned map: 6.3x (120 pairs)
- Test 2 learned map, unseen hospitals: 0.8x (100 pairs)
- Test 2 learned map, same hospital: 1.1x
- Test 1 scrambled, 0 known pairs: 1.4x
- Test 1 scrambled, 50 known pairs: 7.6x
- Test 1 after noise: 1.3x (sigma 2, acc 0.78->0.56)
- Test 1 after removing identity dirs: 8.5x (k=100, acc 0.77)

## PathGen-CLIP (200 patients)
- Test 1 slide -> slide, all: 6.5% vs 0.5% chance (13.0x, 95% CI 3.8%-10.8%, p=0.0005, n=200)
- Test 1 slide -> slide, same diagnosis: 7.5% vs 1.0% chance (7.5x, 95% CI 4.6%-12.0%, p=0.0005, n=200)
- Test 1 slide -> slide, same diagnosis + hospital: 30.9% vs 17.0% chance (1.8x, 95% CI 24.7%-37.8%, p=0.0005, n=188)
- Test 2 report -> slide, zero-shot, all: 0.5% vs 0.5% chance (1.0x, 95% CI 0.1%-2.8%, p=0.6307, n=200)
- Test 2 report -> slide, zero-shot, same diagnosis: 1.0% vs 1.0% chance (1.0x, 95% CI 0.3%-3.6%, p=0.5922, n=200)
- Test 2 report -> slide, zero-shot, same diagnosis + hospital: 17.6% vs 17.0% chance (1.0x, 95% CI 12.8%-23.6%, p=0.4568, n=188)
- Test 2 learned map: 7.3x (120 pairs)
- Test 2 learned map, unseen hospitals: 2.4x (100 pairs)
- Test 2 learned map, same hospital: 1.1x
- Test 1 scrambled, 0 known pairs: 1.2x
- Test 1 scrambled, 50 known pairs: 7.2x
- Test 1 after noise: 1.5x (sigma 2, acc 0.78->0.51)
- Test 1 after removing identity dirs: 9.5x (k=100, acc 0.77)

## QuiltNet (200 patients)
- Test 1 slide -> slide, all: 8.5% vs 0.5% chance (17.0x, 95% CI 5.4%-13.2%, p=0.0005, n=200)
- Test 1 slide -> slide, same diagnosis: 9.0% vs 1.0% chance (9.0x, 95% CI 5.8%-13.8%, p=0.0005, n=200)
- Test 1 slide -> slide, same diagnosis + hospital: 28.2% vs 17.0% chance (1.7x, 95% CI 22.2%-35.0%, p=0.0005, n=188)
- Test 2 report -> slide, zero-shot, all: 0.5% vs 0.5% chance (1.0x, 95% CI 0.1%-2.8%, p=0.6307, n=200)
- Test 2 report -> slide, zero-shot, same diagnosis: 1.0% vs 1.0% chance (1.0x, 95% CI 0.3%-3.6%, p=0.5922, n=200)
- Test 2 report -> slide, zero-shot, same diagnosis + hospital: 16.0% vs 17.0% chance (0.9x, 95% CI 11.4%-21.9%, p=0.6772, n=188)
- Test 2 learned map: 5.3x (120 pairs)
- Test 2 learned map, unseen hospitals: 1.0x (100 pairs)
- Test 2 learned map, same hospital: 1.0x
- Test 1 scrambled, 0 known pairs: 1.2x
- Test 1 scrambled, 50 known pairs: 5.6x
- Test 1 after noise: 1.7x (sigma 1.5, acc 0.79->0.50)
- Test 1 after removing identity dirs: 8.5x (k=100, acc 0.78)

## PLIP (200 patients)
- Test 1 slide -> slide, all: 7.0% vs 0.5% chance (14.0x, 95% CI 4.2%-11.4%, p=0.0005, n=200)
- Test 1 slide -> slide, same diagnosis: 8.5% vs 1.0% chance (8.5x, 95% CI 5.4%-13.2%, p=0.0005, n=200)
- Test 1 slide -> slide, same diagnosis + hospital: 27.7% vs 17.0% chance (1.6x, 95% CI 21.8%-34.5%, p=0.0005, n=188)
- Test 2 report -> slide, zero-shot, all: 0.0% vs 0.5% chance (0.0x, 95% CI 0.0%-1.9%, p=1.0000, n=200)
- Test 2 report -> slide, zero-shot, same diagnosis: 0.0% vs 1.0% chance (0.0x, 95% CI 0.0%-1.9%, p=1.0000, n=200)
- Test 2 report -> slide, zero-shot, same diagnosis + hospital: 17.0% vs 17.0% chance (1.0x, 95% CI 12.3%-23.0%, p=0.5362, n=188)
- Test 2 learned map: 6.3x (120 pairs)
- Test 2 learned map, unseen hospitals: 1.8x (100 pairs)
- Test 2 learned map, same hospital: 1.0x
- Test 1 scrambled, 0 known pairs: 1.2x
- Test 1 scrambled, 50 known pairs: 5.8x
- Test 1 after noise: 1.8x (sigma 1.5, acc 0.77->0.50)
- Test 1 after removing identity dirs: 8.5x (k=100, acc 0.80)
