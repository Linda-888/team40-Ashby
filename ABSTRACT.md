# Tissue Fingerprints: pathology AI embeddings re-identify patients across specimens, and simple defenses fail

*Draft abstract, 10/18. All numbers come from `results/figures/key_numbers.md` (188 audited patients, five models).*

**Background.** Hospitals increasingly share embeddings from pathology foundation models instead of slides, on the
assumption that a vector of numbers is anonymous. We asked whether these embeddings let someone link a patient's
separately released data back together, and whether common defenses prevent it.

**Methods.** We downloaded 200 TCGA lung cancer patients, each with an FFPE diagnostic slide and a frozen-section
slide, plus a pathology report (TCGA-Reports). An audit of slide thumbnails found 12 pairs that were one piece of
tissue under two barcodes (rescanned, or re-cut from the same piece); we removed them, leaving 188 patients (94
LUAD, 94 LUSC, 42 tissue source sites). On NCSA Delta we embedded 200 tissue tiles per slide with five off-the-shelf
image-text models (CONCH, MUSK, PathGen-CLIP, QuiltNet, PLIP). Test 1 asked whether one slide finds the same
patient's other slide. Test 2 asked whether a report finds its patient's slide. Candidates were then restricted
to the same diagnosis and the same tissue source site (hospital). We evaluated three defenses: secret-key rotation
(attacked with Procrustes alignment), additive noise, and projecting out "identity directions" (cross-fitted on
held-out patients).

**Results.** Slide-to-slide re-identification reached 5-8x chance (top-1 2.7-4.3% vs 0.5%), 4-5.5x within the
same diagnosis and 1.4-2.0x within the same hospital (all p<0.01), and was unchanged after removing tiles with pen
ink. Before the audit it appeared as 13-17x. Reports did not find their slides (~1x chance). A linear map learned
from 100 paired examples linked reports to slides at 3-7x, but only through the hospital: within-hospital matching
stayed at 1.0-1.1x. Secret-key rotation was undone with 50 known pairs (5.0-7.4x). Noise pushed Test 1 below 2x
only after subtype classification fell from 0.80-0.88 to 0.49-0.67 balanced accuracy, and removing 50 identity
directions left 2.5-5.5x.

**Conclusion.** Slide embeddings carry an individual tissue fingerprint that survives differences in tissue piece,
preparation, diagnosis and hospital, and that resists simple defenses. Report embeddings leak the hospital rather
than the patient. Public datasets need auditing before such tests: same-tissue pairs inflate re-identification.
Slide embeddings should be governed as identifiable data, and we provide a scorecard a hospital can run before
release.

*Limitations: 188 patients, lung cancer only, five image-text models, full-text OCR reports.*
