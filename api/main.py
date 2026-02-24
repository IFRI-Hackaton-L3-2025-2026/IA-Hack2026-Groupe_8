import json
import os
import pickle
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SUMMARY_CSV = os.path.join(MODELS_DIR, "model_validation_summary.csv")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
INDEX_HTML = os.path.join(STATIC_DIR, "index.html")


def _to_float(value: Any, default: float = 0.5) -> float:
    try:
        out = float(value)
        if np.isnan(out):
            return default
        return out
    except Exception:
        return default


def _load_pickle(path: str) -> Any:
    try:
        with open(path, "rb") as file:
            return pickle.load(file)
    except Exception:
        import joblib

        return joblib.load(path)


def _extract_predictor(obj: Any) -> Any:
    if obj is None:
        return None

    if hasattr(obj, "predict"):
        return obj

    if isinstance(obj, dict):
        for value in obj.values():
            candidate = _extract_predictor(value)
            if candidate is not None:
                return candidate
        return None

    if isinstance(obj, (list, tuple)):
        for value in obj:
            candidate = _extract_predictor(value)
            if candidate is not None:
                return candidate
        return None

    if isinstance(obj, np.ndarray):
        for value in obj.ravel().tolist():
            candidate = _extract_predictor(value)
            if candidate is not None:
                return candidate
        return None

    return None


def _guess_metadata_name(model_file: str) -> list[str]:
    stem = model_file.replace(".pkl", "")
    candidates = [f"{stem}_metadata.json"]

    lower_stem = stem.lower()
    if "random_forest" in lower_stem or "randomforest" in lower_stem:
        candidates.append("randomforest_balanced_metadata.json")
    if "logistic" in lower_stem:
        candidates.append("logistic_regression_balanced_metadata.json")
    if "lgbm" in lower_stem or "lightgbm" in lower_stem:
        candidates.append("lgbm_pipeline_metadata.json")

    return candidates


def _load_best_model_bundle() -> dict[str, Any]:
    if not os.path.exists(SUMMARY_CSV):
        raise RuntimeError(f"Fichier introuvable: {SUMMARY_CSV}")

    summary_df = pd.read_csv(SUMMARY_CSV)
    if summary_df.empty:
        raise RuntimeError("Le fichier model_validation_summary.csv est vide.")

    if "f1_tuned" in summary_df.columns:
        summary_df["_f1_num"] = pd.to_numeric(summary_df["f1_tuned"], errors="coerce")
        summary_df = summary_df.sort_values("_f1_num", ascending=False, na_position="last")

    best_row = summary_df.iloc[0].to_dict()
    model_file = str(best_row.get("model_file", "")).strip()
    if not model_file:
        raise RuntimeError("Aucune colonne model_file exploitable dans le résumé.")

    model_path = os.path.join(MODELS_DIR, model_file)
    if not os.path.exists(model_path):
        raise RuntimeError(f"Modèle introuvable: {model_path}")

    metadata: dict[str, Any] = {}
    for metadata_name in _guess_metadata_name(model_file):
        metadata_path = os.path.join(MODELS_DIR, metadata_name)
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as file:
                metadata = json.load(file)
            break

    model_artifact = _load_pickle(model_path)
    predictor = _extract_predictor(model_artifact)
    if predictor is None:
        raise RuntimeError(f"Aucun estimateur trouvable dans {model_file}.")

    features = metadata.get("features", []) if isinstance(metadata, dict) else []
    if not features:
        raise RuntimeError("Features non trouvées dans les métadonnées du meilleur modèle.")

    threshold = _to_float(best_row.get("threshold_used", 0.5), default=0.5)
    model_label = str(
        best_row.get("model_label")
        or metadata.get("selected_model_name")
        or metadata.get("model_name")
        or model_file.replace(".pkl", "")
    )

    return {
        "model": predictor,
        "model_file": model_file,
        "model_label": model_label,
        "features": features,
        "threshold": threshold,
        "best_row": best_row,
    }


app = FastAPI(title="Predictive Maintenance API", version="1.0.0")
bundle: dict[str, Any] | None = None
startup_error: str | None = None

try:
    bundle = _load_best_model_bundle()
except Exception as error:
    startup_error = str(error)


class PredictRequest(BaseModel):
    instances: list[dict[str, Any]] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    model_file: str
    model_label: str
    threshold: float
    required_features: list[str]
    predictions: list[int]
    probabilities: list[float] | None


@app.get("/health")
def health() -> dict[str, Any]:
    if startup_error is not None:
        return {"status": "degraded", "model_loaded": False, "error": startup_error}
    return {"status": "ok", "model_loaded": True}


@app.get("/")
def ui_home() -> FileResponse:
    if not os.path.exists(INDEX_HTML):
        raise HTTPException(status_code=404, detail="Interface web introuvable.")
    return FileResponse(INDEX_HTML)


def _require_bundle() -> dict[str, Any]:
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Le modèle n'est pas chargé.",
                "error": startup_error or "Erreur de chargement inconnue.",
            },
        )
    return bundle


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    model_bundle = _require_bundle()
    return {
        "model_file": model_bundle["model_file"],
        "model_label": model_bundle["model_label"],
        "threshold": model_bundle["threshold"],
        "required_features": model_bundle["features"],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    model_bundle = _require_bundle()
    model = model_bundle["model"]
    required_features = model_bundle["features"]
    threshold = model_bundle["threshold"]

    input_df = pd.DataFrame(payload.instances)

    missing = [column for column in required_features if column not in input_df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Features manquantes dans le payload.",
                "missing_features": missing,
                "required_features": required_features,
            },
        )

    x_data = input_df[required_features].copy()

    probabilities: list[float] | None = None
    try:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(x_data)
            if isinstance(y_prob, np.ndarray) and y_prob.ndim == 2 and y_prob.shape[1] >= 2:
                probabilities = y_prob[:, 1].astype(float).tolist()
                predictions = (np.array(probabilities) >= threshold).astype(int).tolist()
            else:
                predictions = model.predict(x_data).astype(int).tolist()
        else:
            predictions = model.predict(x_data).astype(int).tolist()
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {error}") from error

    return PredictResponse(
        model_file=model_bundle["model_file"],
        model_label=model_bundle["model_label"],
        threshold=threshold,
        required_features=required_features,
        predictions=predictions,
        probabilities=probabilities,
    )
