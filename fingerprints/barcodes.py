"""TCGA slide barcodes, e.g. TCGA-05-4244-01Z-00-DX1.

    TCGA-05-4244 -01      Z     -00      -DX       1
    patient      sample   vial   portion  slide     number
    (05 = site)  01 tumor        kind: DX = FFPE diagnostic, TS/BS/MS = frozen top/bottom/middle
                 11 normal
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

SLIDE_RE = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})-(\d{2})([A-Z])-(\d{2})-([A-Z]{2})(\d+)")
PROJECT_RE = re.compile(r"TCGA-[A-Z]+")  # TCGA-LUAD; patient barcodes contain digits so never match

TUMOR = {"01"}   # primary solid tumor
NORMAL = {"11"}  # solid tissue normal
FROZEN_KINDS = {"TS", "BS", "MS"}


@dataclass(frozen=True)
class SlideBarcode:
    barcode: str
    patient: str
    site: str
    sample_type: str
    vial: str
    portion: str
    kind: str
    number: int

    @property
    def is_tumor(self) -> bool:
        return self.sample_type in TUMOR

    @property
    def is_normal(self) -> bool:
        return self.sample_type in NORMAL

    @property
    def ffpe(self) -> bool:
        return self.kind == "DX"

    @property
    def frozen(self) -> bool:
        return self.kind in FROZEN_KINDS

    @property
    def specimen(self) -> tuple:
        """Slides with the same key may be neighboring cuts of one block: never pair them.

        FFPE slides of a sample are lumped together because DX1/DX2 can come from the same block.
        Frozen TS/BS of one portion are the top and bottom of the same piece.
        """
        if self.ffpe:
            return (self.patient, self.sample_type, "FFPE")
        return (self.patient, self.sample_type, self.vial, self.portion, "FROZEN")


def parse(name: str) -> Optional[SlideBarcode]:
    m = SLIDE_RE.search(name)
    if not m:
        return None
    patient, sample, vial, portion, kind, number = m.groups()
    return SlideBarcode(
        barcode=m.group(0), patient=patient, site=patient.split("-")[1], sample_type=sample,
        vial=vial, portion=portion, kind=kind, number=int(number),
    )


def project_from_path(path: str) -> Optional[str]:
    """TCGA project from a directory name like .../TCGA-LUAD/..., if the tree is organized that way."""
    for part in reversed(re.split(r"[\\/]", path)):
        if PROJECT_RE.fullmatch(part):
            return part
    return None


def patient_id(text: str) -> Optional[str]:
    """First TCGA patient barcode (TCGA-XX-XXXX) in a string, e.g. a report file name."""
    m = re.search(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", str(text).upper())
    return m.group(0) if m else None


FROZEN_ORDER = {"TS": 0, "BS": 1, "MS": 2}


def choose_pair(slides, mode: str):
    """slides: list of (SlideBarcode, anything) for one patient -> (A, B) or None.

    A is the tumor slide (FFPE diagnostic when there is one); B is a different physical specimen.
    """
    dx = sorted((s for s in slides if s[0].is_tumor and s[0].ffpe), key=lambda s: s[0].number)
    frozen = sorted((s for s in slides if s[0].is_tumor and s[0].frozen),
                    key=lambda s: (FROZEN_ORDER[s[0].kind], s[0].vial, s[0].portion, s[0].number))
    normal = sorted((s for s in slides if s[0].is_normal),
                    key=lambda s: (not s[0].ffpe, s[0].vial, s[0].portion, s[0].number))
    if mode == "dx_vs_frozen" and dx and frozen:
        a, b = dx[0], frozen[0]
    elif mode == "tumor_vs_normal" and (dx or frozen) and normal:
        a, b = (dx or frozen)[0], normal[0]
    else:
        return None
    assert a[0].specimen != b[0].specimen
    return a, b
