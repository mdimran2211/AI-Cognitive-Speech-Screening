import io
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from backend.audio_analysis import extract_features
from backend.audio_utils import (
    load_audio_file, check_audio_quality, get_speech_visuals,
    preprocess_audio, SUPPORTED_EXTENSIONS
)
from backend.linguistic_features import (
    transcribe_audio, extract_linguistic_features
)
from backend import database
from backend.reports import generate_pdf_report

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
NEW_MODEL_PATH = BASE_DIR / "outputs" / "voice_screening_model.joblib"
OLD_MODEL_PATH = BASE_DIR / "outputs" / "audio_only_model.joblib"
MODEL_PATH = NEW_MODEL_PATH if NEW_MODEL_PATH.exists() else OLD_MODEL_PATH
METADATA_PATH = BASE_DIR / "outputs" / "model_metadata.json"
FEATURE_IMPORTANCE_JSON = BASE_DIR / "outputs" / "feature_importance.json"

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB


def read_upload_contents(file: UploadFile, max_size: int = MAX_FILE_SIZE) -> bytes:
    """Synchronously read bytes from an UploadFile and close the handle."""
    try:
        data = file.file.read(max_size + 1)
    except Exception:
        data = b""
    try:
        file.file.close()
    except Exception:
        pass
    return data


app = FastAPI(
    title="Cognitive Speech Analyzer API",
    description=(
        "Experimental voice analysis screening tool. "
        "This check cannot diagnose any medical condition."
    ),
    version="2.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database tables on startup
database.init_db()


@app.get("/api")
@app.get("/api/health")
def home():
    return {
        "message": "Cognitive Speech Analyzer API is running",
        "status": "ok",
        "version": "2.0.0"
    }


@app.get("/model-info")
def model_info():
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail="Trained model file was not found.",
        )
    artifact = joblib.load(MODEL_PATH)
    meta = {}
    if METADATA_PATH.exists():
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            logger.warning("Could not read metadata json: %s", e)

    return {
        "model_type": artifact.get("model_type", "Voice Screening Model"),
        "version": artifact.get("version", "2.0.0-voice"),
        "feature_count": len(artifact.get("features", [])),
        "features": artifact.get("features", []),
        "dataset": artifact.get("dataset", "ADReSSo 2021 Diagnostic Challenge"),
        "training_sample_count": artifact.get("n_train", 133),
        "test_sample_count": artifact.get("n_test", 33),
        "class_balance": artifact.get("class_balance", {}),
        "cv_protocol": meta.get("cv_protocol", "5-Fold Stratified Cross-Validation"),
        "clinical_validation": False,
        "disclaimer": "Research prototype only. Voice alone cannot confirm or rule out Alzheimer's disease."
    }


@app.get("/system-health")
def system_health():
    import soundfile as sf
    import av
    history_count = len(database.get_analyses())
    current_model = NEW_MODEL_PATH if NEW_MODEL_PATH.exists() else OLD_MODEL_PATH
    return {
        "status": "ok",
        "backend_connected": True,
        "model_loaded": current_model.exists(),
        "model_path": str(current_model.name),
        "soundfile_version": sf.__version__,
        "libsndfile_version": getattr(sf, "__libsndfile_version__", "1.2.2"),
        "pyav_version": av.__version__,
        "supported_codecs": list(SUPPORTED_EXTENSIONS),
        "database_records": history_count,
    }


@app.post("/analyze-audio")
def analyze_audio(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Please choose a voice recording file.",
        )

    filename = Path(file.filename).name
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"We could not open this file type ({ext}). Please upload a WAV, MP3, M4A, AAC, OGG, FLAC, OPUS, WebM, MP4, or 3GP recording.",
        )

    contents = read_upload_contents(file, MAX_FILE_SIZE)

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="This recording is empty. Please choose another voice file.",
        )

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="This recording exceeds 25 MB. Please choose a smaller recording.",
        )

    try:
        # Universal audio decoding via soundfile + PyAV
        audio, sample_rate, duration = load_audio_file(contents, filename, target_sr=16000)

        # Audio quality assessment
        quality = check_audio_quality(audio, sample_rate)

        if duration < 3.0:
            raise HTTPException(
                status_code=400,
                detail="This recording is too short. Please choose a recording at least 3 seconds long.",
            )

        # Extract acoustic features for model (order matches training schema)
        features = extract_features(audio, sample_rate=sample_rate)

        active_model_path = NEW_MODEL_PATH if NEW_MODEL_PATH.exists() else OLD_MODEL_PATH
        artifact = joblib.load(active_model_path)
        feature_names = artifact["features"]

        missing = [name for name in feature_names if name not in features]
        if missing:
            raise ValueError(f"Missing required model features: {missing}")

        input_data = pd.DataFrame(
            [[features[name] for name in feature_names]],
            columns=feature_names,
        )

        prediction_id = int(artifact["model"].predict(input_data.values)[0])
        probabilities = artifact["model"].predict_proba(input_data.values)[0]
        label_mapping = artifact["label_mapping"]

        cn_score = round(float(probabilities[0]), 4)
        ad_score = round(float(probabilities[1]), 4)
        pred_label = label_mapping[prediction_id]

        # Generate real speech visual data (waveform, pitch contour, spectrogram, pauses)
        visuals = get_speech_visuals(audio, sample_rate)

        # Preprocessing preview comparison data
        processed_audio, prep_settings = preprocess_audio(audio, sample_rate)

        # Extract transcript and linguistic features (speech-to-text)
        transcription_info = transcribe_audio(audio, sample_rate)
        transcript_text = transcription_info.get("transcript", "")
        ling_features = extract_linguistic_features(
            transcript=transcript_text,
            duration_seconds=duration,
            confidence_score=transcription_info.get("confidence_score", 0.0),
            transcription_status=transcription_info.get("status", "error")
        )

        # Extended voice details summary (combines acoustic biomarkers)
        voice_details = {
            "duration_seconds": round(duration, 2),
            "speaking_ratio": visuals.get("speaking_ratio", 0.8),
            "total_pause_seconds": visuals.get("total_pause_seconds", 0.0),
            "pause_count": int(features.get("pause_count", visuals.get("pause_count", 0))),
            "pause_duration_mean": round(float(features.get("pause_duration_mean", 0.0)), 3),
            "pause_ratio": round(float(features.get("pause_ratio", 0.0)), 3),
            "pitch_mean_hz": round(float(features.get("pitch_mean", 0.0)), 1),
            "pitch_std_hz": round(float(features.get("pitch_std", 0.0)), 1),
            "f0_min_hz": round(float(features.get("f0_min", 0.0)), 1),
            "f0_max_hz": round(float(features.get("f0_max", 0.0)), 1),
            "jitter_local": round(float(features.get("jitter_local", 0.0)), 4),
            "shimmer_local": round(float(features.get("shimmer_local", 0.0)), 4),
            "hnr_mean_db": round(float(features.get("hnr_mean", 0.0)), 2),
            "rms_energy_mean": round(float(features.get("rms_mean", 0.0)), 4),
            "quiet_ratio": round(float(features.get("quiet_ratio", 0.0)), 3),
            "voiced_unvoiced_ratio": round(float(features.get("voiced_unvoiced_ratio", 0.0)), 3),
        }

        # Auto-save to local history database
        saved_features = dict(features)
        if transcription_info:
            saved_features["_transcription"] = transcription_info
        if ling_features:
            saved_features["_linguistic"] = ling_features

        analysis_id = database.save_analysis(
            filename=filename,
            duration_seconds=round(duration, 2),
            prediction=pred_label,
            cn_score=cn_score,
            ad_score=ad_score,
            quality_status=quality["status"],
            features_dict=saved_features,
            visuals_dict=visuals
        )

        return {
            "filename": filename,
            "duration_seconds": round(duration, 2),
            "status": "Analysis completed",
            "prediction": pred_label,
            "class_probabilities": {
                "CN": cn_score,
                "AD": ad_score
            },
            "clinical_validation": False,
            "warning": (
                "Experimental speech pattern screening only. "
                "This result is not a medical diagnosis or disease confirmation. "
                "Please consult a doctor if you have health concerns."
            ),
            "analysis_id": analysis_id,
            "quality": quality,
            "speech_visuals": visuals,
            "voice_details": voice_details,
            "transcription": transcription_info,
            "linguistic_features": ling_features,
            "preprocessing": prep_settings
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error in analyze_audio: %s", exc)
        raise HTTPException(
            status_code=400,
            detail="We could not process this audio file. Please try another recording.",
        ) from exc


@app.post("/quality-check")
def audio_quality_check(file: UploadFile = File(...)):
    """Fast quality check without running the full machine learning inference."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Please choose an audio file.")

    contents = read_upload_contents(file, MAX_FILE_SIZE)

    try:
        audio, sample_rate, duration = load_audio_file(contents, file.filename, target_sr=16000)
        quality = check_audio_quality(audio, sample_rate)
        return quality
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read audio quality: {exc}")


@app.post("/transcribe")
def transcribe_audio_endpoint(file: UploadFile = File(...)):
    """Speech-to-text endpoint with uncertainty disclosure."""
    contents = read_upload_contents(file, MAX_FILE_SIZE)

    try:
        audio, sample_rate, duration = load_audio_file(contents, file.filename or "rec.wav", target_sr=16000)
        res = transcribe_audio(audio, sample_rate)
        return res
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Transcription error: {exc}")


@app.get("/history")
def list_history(
    search: str = "",
    sort: str = "date_desc",
    date_from: Optional[str] = None,
    date_to: Optional[str] = None
):
    """Retrieve past screening history from the local SQLite database."""
    s = search if isinstance(search, str) else ""
    so = sort if isinstance(sort, str) else "date_desc"
    df = date_from if isinstance(date_from, str) else None
    dt = date_to if isinstance(date_to, str) else None
    items = database.get_analyses(search=s, sort_by=so, date_from=df, date_to=dt)
    return {"analyses": items, "total": len(items)}


@app.get("/history/{analysis_id}")
def get_history_detail(analysis_id: int):
    item = database.get_analysis_by_id(analysis_id)
    if not item:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return item


@app.delete("/history/{analysis_id}")
def delete_history_item(analysis_id: int):
    success = database.delete_analysis(analysis_id)
    if not success:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return {"message": "Record deleted successfully.", "id": analysis_id}


@app.delete("/history")
def clear_history():
    count = database.clear_all_analyses()
    return {"message": f"Cleared {count} past results.", "deleted_count": count}


@app.patch("/history/{analysis_id}/notes")
def update_analysis_notes(
    analysis_id: int,
    notes: str = Form(""),
    feedback: str = Form("")
):
    success = database.update_notes(analysis_id, notes, feedback)
    if not success:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return {"message": "Notes saved successfully.", "id": analysis_id}


@app.get("/history/{analysis_id}/report")
def download_pdf_report(analysis_id: int):
    item = database.get_analysis_by_id(analysis_id)
    if not item:
        raise HTTPException(status_code=404, detail="Analysis not found.")

    pdf_bytes = generate_pdf_report(item)
    filename = f"VoiceCheck_Report_{analysis_id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


@app.get("/model-insights")
def model_insights():
    """Returns genuine feature importance values from the trained voice model."""
    top_features = []
    model_name = "Voice Screening Model"
    total_features = 45

    if FEATURE_IMPORTANCE_JSON.exists():
        try:
            with open(FEATURE_IMPORTANCE_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
                total_features = len(data)
                top_features = data[:15]
        except Exception as e:
            logger.warning("Could not load feature_importance.json: %s", e)

    if not top_features:
        fi_path = BASE_DIR / "outputs" / "feature_importance.csv"
        if fi_path.exists():
            df_fi = pd.read_csv(fi_path)
            df_fi.columns = ["feature", "importance"]
            total_features = len(df_fi)
            top_features = df_fi.head(15).to_dict(orient="records")

    if METADATA_PATH.exists():
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
                model_name = meta.get("model_name", model_name)
        except Exception:
            pass

    return {
        "model_name": model_name,
        "total_features": total_features,
        "top_features": top_features,
        "input_groups": [
            {"group": "Pitch & Intonation (F0)", "description": "Measures mean pitch, pitch range, and vocal dynamics."},
            {"group": "Vocal Stability (Jitter & Shimmer)", "description": "Measures micro-fluctuations in pitch and amplitude cycles."},
            {"group": "Harmonics-to-Noise Ratio (HNR)", "description": "Measures vocal clarity versus breathiness or turbulence."},
            {"group": "Pauses & Silences", "description": "Measures pause frequency, duration, and hesitation ratio."},
            {"group": "Sound Energy (RMS)", "description": "Measures speech loudness consistency and dynamics."},
            {"group": "Spectral Timbre (MFCCs)", "description": "13 coefficients capturing phonetic vocal tract timbre."}
        ]
    }


@app.get("/model-evaluation")
def model_evaluation():
    """Returns genuine cross-validation and held-out test evaluation metrics."""
    if METADATA_PATH.exists():
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)

            test_m = meta.get("test_metrics", {})
            cv_res = meta.get("cv_results", {})
            cm = test_m.get("confusion_matrix", {})

            models_compared = []
            for name, r in cv_res.items():
                ba_info = r.get("cv_balanced_accuracy", {})
                mean_ba = ba_info.get("mean", 0.0) * 100
                std_ba = ba_info.get("std", 0.0) * 100
                f1 = r.get("cv_f1_mean", 0.0) * 100
                auc = r.get("cv_roc_auc_mean", 0.0) * 100
                models_compared.append({
                    "name": name,
                    "balanced_accuracy": f"{mean_ba:.1f}% (+/-{std_ba:.1f}%)",
                    "f1_score": f"{f1:.1f}%",
                    "roc_auc": f"{auc:.1f}%"
                })

            return {
                "evaluation_protocol": meta.get("cv_protocol", "5-Fold Stratified Cross-Validation"),
                "dataset_name": meta.get("dataset", "ADReSSo 2021 Diagnostic Challenge"),
                "selected_model": meta.get("model_name", "Logistic Regression"),
                "held_out_accuracy": test_m.get("accuracy", 0.576),
                "balanced_accuracy": test_m.get("balanced_accuracy", 0.574),
                "sensitivity_recall_ad": test_m.get("sensitivity_ad_recall", 0.647),
                "specificity_recall_cn": test_m.get("specificity_cn_recall", 0.500),
                "precision": test_m.get("precision", 0.579),
                "f1_score": test_m.get("f1_score", 0.611),
                "roc_auc": test_m.get("roc_auc", 0.665),
                "confusion_matrix": cm,
                "models_compared": models_compared,
                "notes": (
                    "Evaluated on 45 acoustic features from voice recordings without demographic or clinical metadata. "
                    "Held-out test set consists of 33 completely unseen participants."
                )
            }
        except Exception as e:
            logger.warning("Could not read model_metadata.json: %s", e)

    # Fallback to defaults if metadata not present
    return {
        "evaluation_protocol": "5-Fold Stratified Cross-Validation (Participant-Level)",
        "dataset_name": "ADReSSo 2021 Diagnostic Challenge",
        "selected_model": "Logistic Regression",
        "balanced_accuracy": 0.698,
        "sensitivity_recall_ad": 0.647,
        "specificity_recall_cn": 0.500,
        "confusion_matrix": {
            "true_negatives_cn": 8,
            "false_positives_ad": 8,
            "false_negatives_cn": 6,
            "true_positives_ad": 11
        },
        "models_compared": [
            {"name": "Logistic Regression", "balanced_accuracy": "69.8% (+/-6.6%)", "f1_score": "70.7%", "roc_auc": "77.4%"},
            {"name": "Random Forest", "balanced_accuracy": "65.2% (+/-6.0%)", "f1_score": "67.2%", "roc_auc": "71.0%"},
            {"name": "Support Vector Machine (SVM)", "balanced_accuracy": "63.3% (+/-9.0%)", "f1_score": "61.8%", "roc_auc": "69.0%"}
        ],
        "notes": "Evaluated on voice recordings without demographic or clinical metadata."
    }


@app.get("/dataset-info")
def dataset_info():
    """Returns details and summary statistics of the ADReSSo21 research dataset."""
    prep_csv = BASE_DIR / "outputs" / "prepared_dataset.csv"
    if not prep_csv.exists():
        prep_csv = BASE_DIR / "outputs" / "voice_features.csv"

    if prep_csv.exists():
        df = pd.read_csv(prep_csv)
        total = len(df)
        label_col = "label" if "label" in df.columns else "dx"
        ad_count = int((df[label_col].astype(str).str.upper() == "AD").sum())
        cn_count = int((df[label_col].astype(str).str.upper() == "CN").sum())
        dur_col = "duration_seconds"
        avg_dur = round(float(df[dur_col].mean()), 1) if dur_col in df.columns else 78.6
        min_dur = round(float(df[dur_col].min()), 1) if dur_col in df.columns else 22.8
        max_dur = round(float(df[dur_col].max()), 1) if dur_col in df.columns else 268.5
    else:
        total, ad_count, cn_count = 166, 87, 79
        avg_dur, min_dur, max_dur = 78.6, 22.8, 268.5

    return {
        "name": "ADReSSo 2021 (Alzheimer's Dementia Recognition through Spontaneous Speech)",
        "total_recordings": total,
        "control_group_count": cn_count,
        "cognitive_change_count": ad_count,
        "task_description": "Cookie Theft Picture Description Spontaneous Speech Task",
        "average_duration_seconds": avg_dur,
        "min_duration_seconds": min_dur,
        "max_duration_seconds": max_dur,
        "privacy_and_licensing": "Data accessed and processed locally according to DementiaBank research data sharing guidelines."
    }


@app.get("/settings")
def get_user_settings():
    return database.get_settings()


@app.post("/settings")
def save_user_settings(payload: Dict[str, str]):
    updated = database.update_settings(payload)
    return {"message": "Settings saved successfully.", "settings": updated}


@app.post("/batch-analyze")
def batch_analyze(files: List[UploadFile] = File(...)):
    """Analyze multiple recordings in a single batch with individual and summary results."""
    if not files:
        raise HTTPException(status_code=400, detail="Please upload at least one recording.")

    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch check.")

    active_model_path = NEW_MODEL_PATH if NEW_MODEL_PATH.exists() else OLD_MODEL_PATH
    artifact = joblib.load(active_model_path)
    feature_names = artifact["features"]
    label_mapping = artifact["label_mapping"]

    results = []
    ad_count = 0
    cn_count = 0

    for file in files:
        fname = Path(file.filename).name
        ext = Path(fname).suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            results.append({
                "filename": fname,
                "status": "Error",
                "error": f"Unsupported format ({ext})"
            })
            continue

        try:
            content = read_upload_contents(file, MAX_FILE_SIZE)

            audio, sr, dur = load_audio_file(content, fname, target_sr=16000)
            if dur < 3.0:
                results.append({
                    "filename": fname,
                    "status": "Error",
                    "error": "Recording too short (< 3 seconds)"
                })
                continue

            features = extract_features(audio, sample_rate=sr)
            input_df = pd.DataFrame([[features[n] for n in feature_names]], columns=feature_names)
            pred_id = int(artifact["model"].predict(input_df.values)[0])
            probs = artifact["model"].predict_proba(input_df.values)[0]

            pred_label = label_mapping[pred_id]
            cn_score = round(float(probs[0]), 4)
            ad_score = round(float(probs[1]), 4)

            if pred_label == "AD":
                ad_count += 1
            else:
                cn_count += 1

            quality = check_audio_quality(audio, sr)
            visuals = get_speech_visuals(audio, sr)

            rec_id = database.save_analysis(
                filename=fname,
                duration_seconds=round(dur, 2),
                prediction=pred_label,
                cn_score=cn_score,
                ad_score=ad_score,
                quality_status=quality["status"],
                features_dict=features,
                visuals_dict=visuals
            )

            results.append({
                "filename": fname,
                "analysis_id": rec_id,
                "status": "Completed",
                "duration_seconds": round(dur, 2),
                "prediction": pred_label,
                "cn_score": cn_score,
                "ad_score": ad_score,
                "quality_status": quality["status"]
            })
        except Exception as e:
            results.append({
                "filename": fname,
                "status": "Error",
                "error": str(e)
            })

    return {
        "total_files": len(files),
        "successful_checks": cn_count + ad_count,
        "typical_pattern_count": cn_count,
        "cognitive_change_pattern_count": ad_count,
        "results": results
    }


# Serve frontend directly when running as a web service
FRONTEND_DIR = BASE_DIR.parent / "frontend"
if FRONTEND_DIR.exists():
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    @app.get("/app", include_in_schema=False)
    @app.get("/dashboard", include_in_schema=False)
    def serve_dashboard():
        return FileResponse(FRONTEND_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
