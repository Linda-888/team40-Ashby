# Tissue Fingerprints: pathology AI embeddings re-identify patients across specimens, and simple defenses fail

*Draft abstract, 10/18. All numbers come from `results/figures/key_numbers.md` (200 patients, five models).*

**Background.** Hospitals increasingly share embeddings from pathology foundation models instead of slides, on the
assumption that a vector of numbers is anonymous. We asked whether these embeddings let someone link a patient's
separately released data back together, and whether common defenses prevent it.

**Methods.** We used 200 TCGA lung cancer patients (100 LUAD, 100 LUSC, 44 tissue source sites). Each had two
physically different specimens, an FFPE diagnostic slide and a frozen-section slide, plus a pathology report
(TCGA-Reports). On NCSA Delta we embedded 200 tissue tiles per slide with five image-text models (CONCH,
MUSK, PathGen-CLIP, QuiltNet, PLIP). Test 1 asked whether one slide finds the same patient's other specimen. Test 2 asked
whether a report finds its patient's slide. Candidates were then restricted to the same diagnosis and the same
tissue source site (hospital). We evaluated three defenses: secret-key rotation (attacked with Procrustes
alignment), additive noise, and projecting out "identity directions" (cross-fitted on held-out patients).

**Results.** Slide-to-slide re-identification reached 13-17x chance (top-1 6.5-8.5% vs 0.5%). It was 7.5-9x within
the same diagnosis and 1.6-2.2x within the same hospital (all p<0.001). Reports did not find their slides (~1x chance).
A linear map learned from 120 paired examples linked reports to slides at 4-7x, but only through the hospital:
within-hospital matching stayed at 1.0-1.1x. Secret-key rotation was undone with 50 known pairs (5.6-8.8x). Noise
pushed Test 1 below 2x only once subtype classification fell to about chance (balanced accuracy 0.50-0.56), and
removing 100 identity directions left 8.5-13x.

**Conclusion.** Slide embeddings carry an individual tissue fingerprint that survives differences in specimen,
preparation, diagnosis and hospital, and that resists simple defenses. Report embeddings leak the hospital rather
than the patient. Slide embeddings should be governed as identifiable data, and we provide a scorecard a hospital
can run before release.

*Limitations: five image-text models, lung cancer only, 200 patients, full-text OCR reports.*
