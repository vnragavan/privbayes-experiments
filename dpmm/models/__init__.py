# PrivBayes-only: only export PrivBayesGM (no AIM/MST)
import json
from pathlib import Path

from dpmm.models.base.base import GenerativeModel
from dpmm.models.priv_bayes import PrivBayesGM

MODELS = [PrivBayesGM]
MODEL_DICT = {MODEL.name: MODEL for MODEL in MODELS}


def load_model(path: Path) -> GenerativeModel:
    with (path / "model_type.json").open("r") as fr:
        model_type = json.load(fr)
    if model_type not in MODEL_DICT:
        raise ValueError(f"Unknown model type {model_type}; this copy only supports PrivBayes.")
    return MODEL_DICT[model_type].load(path)
