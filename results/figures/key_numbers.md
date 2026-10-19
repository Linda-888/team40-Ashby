# Key numbers (re-identification as top-1, x chance)

## Across models (x chance)

- Test 1 slide -> slide: 5.0x to 8.0x  (CONCH 7.0x, MUSK 5.0x, PathGen-CLIP 8.0x, QuiltNet 7.0x, PLIP 7.0x)
- Test 1 same hospital: 1.4x to 2.0x  (CONCH 2.0x, MUSK 1.8x, PathGen-CLIP 1.6x, QuiltNet 1.4x, PLIP 1.4x)
- Test 2 report -> slide, zero-shot: 0.0x to 1.0x  (CONCH 0.0x, MUSK 1.0x, PathGen-CLIP 1.0x, QuiltNet 1.0x, PLIP 0.0x)
- Test 2 learned map: 3.4x to 7.1x  (CONCH 3.4x, MUSK 5.9x, PathGen-CLIP 5.6x, QuiltNet 5.0x, PLIP 7.1x)
- Test 2 learned map, unseen hospitals: 1.2x to 1.6x  (CONCH 1.4x, MUSK 1.3x, PathGen-CLIP 1.4x, QuiltNet 1.2x, PLIP 1.6x)
- Test 2 learned map, same hospital: 1.0x to 1.1x  (CONCH 1.0x, MUSK 1.1x, PathGen-CLIP 1.1x, QuiltNet 1.1x, PLIP 1.1x)
- Test 1 scrambled, 0 known pairs: 0.8x to 1.2x  (CONCH 0.8x, MUSK 1.2x, PathGen-CLIP 1.2x, QuiltNet 1.0x, PLIP 1.0x)
- Test 1 scrambled, 50 known pairs: 5.0x to 7.4x  (CONCH 5.0x, MUSK 7.4x, PathGen-CLIP 6.8x, QuiltNet 6.2x, PLIP 6.4x)
- Test 1 after noise: 1.0x to 1.8x  (CONCH 1.5x, MUSK 1.0x, PathGen-CLIP 1.7x, QuiltNet 1.8x, PLIP 1.8x)
- Test 1 after removing identity dirs: 2.5x to 5.5x  (CONCH 5.0x, MUSK 5.5x, PathGen-CLIP 2.5x, QuiltNet 4.0x, PLIP 3.0x)

## CONCH (188 patients)
- Test 1 slide -> slide, all: 3.7% vs 0.5% chance (7.0x, 95% CI 1.8%-7.5%, p=0.0010, n=188)
- Test 1 slide -> slide, same diagnosis: 5.9% vs 1.1% chance (5.5x, 95% CI 3.3%-10.2%, p=0.0005, n=188)
- Test 1 slide -> slide, same diagnosis + hospital: 36.0% vs 18.0% chance (2.0x, 95% CI 29.3%-43.2%, p=0.0005, n=178)
- Test 2 report -> slide, zero-shot, all: 0.0% vs 0.5% chance (0.0x, 95% CI 0.0%-2.0%, p=1.0000, n=188)
- Test 2 report -> slide, zero-shot, same diagnosis: 0.5% vs 1.1% chance (0.5x, 95% CI 0.1%-3.0%, p=0.8626, n=188)
- Test 2 report -> slide, zero-shot, same diagnosis + hospital: 19.1% vs 18.0% chance (1.1x, 95% CI 14.0%-25.5%, p=0.3803, n=178)
- Test 2 learned map: 3.4x (100 pairs)
- Test 2 learned map, unseen hospitals: 1.4x (100 pairs)
- Test 2 learned map, same hospital: 1.0x
- Test 1 scrambled, 0 known pairs: 0.8x
- Test 1 scrambled, 50 known pairs: 5.0x
- Test 1 after noise: 1.5x (sigma 5, acc 0.88->0.49)
- Test 1 after removing identity dirs: 5.0x (k=50, acc 0.84)

## MUSK (188 patients)
- Test 1 slide -> slide, all: 2.7% vs 0.5% chance (5.0x, 95% CI 1.1%-6.1%, p=0.0040, n=188)
- Test 1 slide -> slide, same diagnosis: 4.3% vs 1.1% chance (4.0x, 95% CI 2.2%-8.2%, p=0.0025, n=188)
- Test 1 slide -> slide, same diagnosis + hospital: 32.6% vs 18.0% chance (1.8x, 95% CI 26.1%-39.8%, p=0.0005, n=178)
- Test 2 report -> slide, zero-shot, all: 0.5% vs 0.5% chance (1.0x, 95% CI 0.1%-3.0%, p=0.6357, n=188)
- Test 2 report -> slide, zero-shot, same diagnosis: 0.5% vs 1.1% chance (0.5x, 95% CI 0.1%-3.0%, p=0.8626, n=188)
- Test 2 report -> slide, zero-shot, same diagnosis + hospital: 22.5% vs 18.0% chance (1.2x, 95% CI 17.0%-29.1%, p=0.0610, n=178)
- Test 2 learned map: 5.9x (100 pairs)
- Test 2 learned map, unseen hospitals: 1.3x (100 pairs)
- Test 2 learned map, same hospital: 1.1x
- Test 1 scrambled, 0 known pairs: 1.2x
- Test 1 scrambled, 50 known pairs: 7.4x
- Test 1 after noise: 1.0x (sigma 1.5, acc 0.80->0.58)
- Test 1 after removing identity dirs: 5.5x (k=50, acc 0.74)

## PathGen-CLIP (188 patients)
- Test 1 slide -> slide, all: 4.3% vs 0.5% chance (8.0x, 95% CI 2.2%-8.2%, p=0.0005, n=188)
- Test 1 slide -> slide, same diagnosis: 5.3% vs 1.1% chance (5.0x, 95% CI 2.9%-9.5%, p=0.0010, n=188)
- Test 1 slide -> slide, same diagnosis + hospital: 29.2% vs 18.0% chance (1.6x, 95% CI 23.0%-36.3%, p=0.0005, n=178)
- Test 2 report -> slide, zero-shot, all: 0.5% vs 0.5% chance (1.0x, 95% CI 0.1%-3.0%, p=0.6357, n=188)
- Test 2 report -> slide, zero-shot, same diagnosis: 1.1% vs 1.1% chance (1.0x, 95% CI 0.3%-3.8%, p=0.5947, n=188)
- Test 2 report -> slide, zero-shot, same diagnosis + hospital: 18.5% vs 18.0% chance (1.0x, 95% CI 13.5%-24.9%, p=0.4473, n=178)
- Test 2 learned map: 5.6x (100 pairs)
- Test 2 learned map, unseen hospitals: 1.4x (100 pairs)
- Test 2 learned map, same hospital: 1.1x
- Test 1 scrambled, 0 known pairs: 1.2x
- Test 1 scrambled, 50 known pairs: 6.8x
- Test 1 after noise: 1.7x (sigma 1.5, acc 0.80->0.51)
- Test 1 after removing identity dirs: 2.5x (k=50, acc 0.78)

## QuiltNet (188 patients)
- Test 1 slide -> slide, all: 3.7% vs 0.5% chance (7.0x, 95% CI 1.8%-7.5%, p=0.0010, n=188)
- Test 1 slide -> slide, same diagnosis: 4.8% vs 1.1% chance (4.5x, 95% CI 2.5%-8.8%, p=0.0010, n=188)
- Test 1 slide -> slide, same diagnosis + hospital: 25.3% vs 18.0% chance (1.4x, 95% CI 19.5%-32.1%, p=0.0065, n=178)
- Test 2 report -> slide, zero-shot, all: 0.5% vs 0.5% chance (1.0x, 95% CI 0.1%-3.0%, p=0.6357, n=188)
- Test 2 report -> slide, zero-shot, same diagnosis: 1.1% vs 1.1% chance (1.0x, 95% CI 0.3%-3.8%, p=0.5947, n=188)
- Test 2 report -> slide, zero-shot, same diagnosis + hospital: 16.9% vs 18.0% chance (0.9x, 95% CI 12.1%-23.0%, p=0.6837, n=178)
- Test 2 learned map: 5.0x (100 pairs)
- Test 2 learned map, unseen hospitals: 1.2x (100 pairs)
- Test 2 learned map, same hospital: 1.1x
- Test 1 scrambled, 0 known pairs: 1.0x
- Test 1 scrambled, 50 known pairs: 6.2x
- Test 1 after noise: 1.8x (sigma 0.5, acc 0.82->0.67)
- Test 1 after removing identity dirs: 4.0x (k=50, acc 0.80)

## PLIP (188 patients)
- Test 1 slide -> slide, all: 3.7% vs 0.5% chance (7.0x, 95% CI 1.8%-7.5%, p=0.0010, n=188)
- Test 1 slide -> slide, same diagnosis: 4.3% vs 1.1% chance (4.0x, 95% CI 2.2%-8.2%, p=0.0025, n=188)
- Test 1 slide -> slide, same diagnosis + hospital: 25.3% vs 18.0% chance (1.4x, 95% CI 19.5%-32.1%, p=0.0065, n=178)
- Test 2 report -> slide, zero-shot, all: 0.0% vs 0.5% chance (0.0x, 95% CI 0.0%-2.0%, p=1.0000, n=188)
- Test 2 report -> slide, zero-shot, same diagnosis: 0.0% vs 1.1% chance (0.0x, 95% CI 0.0%-2.0%, p=1.0000, n=188)
- Test 2 report -> slide, zero-shot, same diagnosis + hospital: 17.4% vs 18.0% chance (1.0x, 95% CI 12.5%-23.7%, p=0.6072, n=178)
- Test 2 learned map: 7.1x (100 pairs)
- Test 2 learned map, unseen hospitals: 1.6x (100 pairs)
- Test 2 learned map, same hospital: 1.1x
- Test 1 scrambled, 0 known pairs: 1.0x
- Test 1 scrambled, 50 known pairs: 6.4x
- Test 1 after noise: 1.8x (sigma 0.75, acc 0.80->0.62)
- Test 1 after removing identity dirs: 3.0x (k=50, acc 0.80)
