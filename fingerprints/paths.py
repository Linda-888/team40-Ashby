"""Where every artifact lives. All paths come from config.sh (via environment variables)."""
import os
from pathlib import Path

MODELS = ("conch", "musk", "pathgen", "quilt", "plip")  # the plan's three, then two ungated backups


def root() -> Path:
    return Path(os.environ.get("TF_ROOT", "."))


def work() -> Path:
    return Path(os.environ.get("TF_WORK", str(root())))


def patients_csv() -> Path:
    return root() / "data" / "patients.csv"


def slides_csv() -> Path:
    return root() / "data" / "slides.csv"


def reports_csv() -> Path:
    return Path(os.environ.get("TF_REPORTS_CSV", str(root() / "data" / "reports.csv")))


def tiles_dir() -> Path:
    return work() / "tiles"


def thumbs_dir() -> Path:
    return root() / "thumbs"


def emb_dir(model: str) -> Path:
    return root() / "emb" / model


def results_dir(model: str) -> Path:
    d = root() / "results" / model
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_atomic(path: Path, write) -> None:
    """Write via a temp file and rename, so a killed job never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        write(f)
    os.replace(tmp, path)
