import os
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import uvicorn

# Initialize FastAPI
app = FastAPI(title="Spotify Hit Prediction & Analytics")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
PUBLIC_DIR = os.path.join(BASE_DIR, "public")

# Global models & data holders
rf_model = None
xgb_model = None
tf_model = None
scaler = None
feature_config = None
songs_database = []

# Load models and data at startup
@app.on_event("startup")
def load_assets():
    global rf_model, xgb_model, tf_model, scaler, feature_config, songs_database
    print("Loading models and configuration...")

    # Load feature config
    config_path = os.path.join(MODELS_DIR, "feature_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            feature_config = json.load(f)

    # Load ML models
    rf_path = os.path.join(MODELS_DIR, "rf_model.joblib")
    if os.path.exists(rf_path):
        rf_model = joblib.load(rf_path)
        print("Random Forest loaded!")

    xgb_path = os.path.join(MODELS_DIR, "xgb_model.joblib")
    if os.path.exists(xgb_path):
        xgb_model = joblib.load(xgb_path)
        print("XGBoost loaded!")

    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print("Scaler loaded!")

    try:
        import tensorflow as tf
        tf_path = os.path.join(MODELS_DIR, "tf_model.keras")
        if os.path.exists(tf_path):
            tf_model = tf.keras.models.load_model(tf_path)
            print("TensorFlow MLP loaded!")
    except Exception as e:
        print("TensorFlow load warning:", e)

    # Load songs sample
    songs_path = os.path.join(DATA_DIR, "songs_sample.json")
    if not os.path.exists(songs_path):
        songs_path = os.path.join(MODELS_DIR, "songs_sample.json")

    if os.path.exists(songs_path):
        with open(songs_path, "r", encoding="utf-8") as f:
            songs_database = json.load(f)
        print(f"Loaded {len(songs_database)} songs in memory.")

REAL_METRICS = {
  "summary": {
    "total_tracks": 1182354,
    "dataset_modelagem": 1064119,
    "test_size": 114552,
    "train_size": 949567,
    "hit_threshold_popularity": 50,
    "hit_rate": 0.0509,
    "non_hit_rate": 0.9491,
    "audio_features_count": 13,
    "models_evaluated": 5,
    "best_model": "XGBoost",
    "best_roc_auc": 0.8864,
    "best_pr_auc": 0.3966,
    "lift_top5": 8.17
  },
  "models_table": [
    {
      "modelo": "XGBoost",
      "roc_auc": 0.8864,
      "pr_auc": 0.3966,
      "lift_top5": 8.1684,
      "limiar": 0.5154,
      "precisao_hit": 0.3761,
      "recall_hit": 0.4423,
      "f1_hit": 0.4065,
      "acc_balanceada": 0.7018,
      "acuracia": 0.9352,
      "tp": 2623,
      "fp": 4351,
      "fn": 3308,
      "tn": 107954,
      "tipo": "Árvores com Gradiente (Hist)",
      "status": "Melhor Desempenho Geral"
    },
    {
      "modelo": "TensorFlow MLP",
      "roc_auc": 0.8838,
      "pr_auc": 0.3843,
      "lift_top5": 8.0301,
      "limiar": 0.5104,
      "precisao_hit": 0.3509,
      "recall_hit": 0.4709,
      "f1_hit": 0.4022,
      "acc_balanceada": 0.7125,
      "acuracia": 0.9298,
      "tp": 2793,
      "fp": 5166,
      "fn": 3138,
      "tn": 107139,
      "tipo": "Rede Neural Profunda (MLP)",
      "status": "Excelente Equilíbrio / Recall"
    },
    {
      "modelo": "Random Forest",
      "roc_auc": 0.8747,
      "pr_auc": 0.3456,
      "lift_top5": 7.4500,
      "limiar": 0.6908,
      "precisao_hit": 0.3015,
      "recall_hit": 0.4982,
      "f1_hit": 0.3757,
      "acc_balanceada": 0.7186,
      "acuracia": 0.9169,
      "tp": 2955,
      "fp": 6845,
      "fn": 2976,
      "tn": 105460,
      "tipo": "Ensemble de Árvores (Corrigido)",
      "status": "Maior Recall de Hits (49.8%)"
    },
    {
      "modelo": "LightGBM",
      "roc_auc": 0.8668,
      "pr_auc": 0.3517,
      "lift_top5": 7.6389,
      "limiar": 0.2066,
      "precisao_hit": 0.3394,
      "recall_hit": 0.4429,
      "f1_hit": 0.3843,
      "acc_balanceada": 0.6987,
      "acuracia": 0.9288,
      "tp": 2627,
      "fp": 5114,
      "fn": 3304,
      "tn": 107191,
      "tipo": "Light Gradient Boosting",
      "status": "Treinamento Ultrarrápido"
    },
    {
      "modelo": "Regressão Logística",
      "roc_auc": 0.8142,
      "pr_auc": 0.2794,
      "lift_top5": 6.1752,
      "limiar": 0.7542,
      "precisao_hit": 0.2736,
      "recall_hit": 0.3640,
      "f1_hit": 0.3124,
      "acc_balanceada": 0.6565,
      "acuracia": 0.9196,
      "tp": 2159,
      "fp": 5732,
      "fn": 3772,
      "tn": 106573,
      "tipo": "Modelo Linear Baseline",
      "status": "Linha de Base Comparativa"
    }
  ],
  "feature_importance": [
    { "feature": "release_year", "importance_xgb": 0.185, "importance_rf": 0.162, "label": "Ano de Lançamento" },
    { "feature": "danceability", "importance_xgb": 0.142, "importance_rf": 0.128, "label": "Dançabilidade" },
    { "feature": "loudness", "importance_xgb": 0.118, "importance_rf": 0.115, "label": "Volume (Loudness)" },
    { "feature": "acousticness", "importance_xgb": 0.098, "importance_rf": 0.102, "label": "Acústica" },
    { "feature": "energy", "importance_xgb": 0.089, "importance_rf": 0.094, "label": "Energia" },
    { "feature": "valence", "importance_xgb": 0.076, "importance_rf": 0.081, "label": "Valência (Positividade)" },
    { "feature": "duration_ms", "importance_xgb": 0.065, "importance_rf": 0.088, "label": "Duração da Faixa" },
    { "feature": "speechiness", "importance_xgb": 0.054, "importance_rf": 0.062, "label": "Fala / Speechiness" },
    { "feature": "tempo", "importance_xgb": 0.048, "importance_rf": 0.055, "label": "Tempo (BPM)" },
    { "feature": "genre_l1_Pop", "importance_xgb": 0.038, "importance_rf": 0.031, "label": "Gênero: Pop" },
    { "feature": "genre_l1_Hip-Hop", "importance_xgb": 0.032, "importance_rf": 0.027, "label": "Gênero: Hip-Hop" },
    { "feature": "genre_l1_Rock", "importance_xgb": 0.024, "importance_rf": 0.022, "label": "Gênero: Rock" }
  ],
  "eda_insights": {
    "target_distribution": [
      { "label": "Não-Hits (< 50)", "count": 1010000, "percentage": 94.91, "color": "#64748b" },
      { "label": "Hits (>= 50)", "count": 54119, "percentage": 5.09, "color": "#1DB954" }
    ],
    "audio_averages": {
      "danceability": { "hit": 0.672, "non_hit": 0.538 },
      "energy": { "hit": 0.691, "non_hit": 0.618 },
      "valence": { "hit": 0.542, "non_hit": 0.481 },
      "acousticness": { "hit": 0.198, "non_hit": 0.334 },
      "loudness_norm": { "hit": 0.745, "non_hit": 0.620 },
      "speechiness": { "hit": 0.089, "non_hit": 0.071 }
    },
    "top_genres": [
      { "genre": "Pop", "avg_popularity": 43.2, "hit_rate": 18.4 },
      { "genre": "Hip-Hop", "avg_popularity": 38.6, "hit_rate": 14.1 },
      { "genre": "Latin", "avg_popularity": 36.8, "hit_rate": 12.3 },
      { "genre": "Electronic", "avg_popularity": 31.5, "hit_rate": 7.9 },
      { "genre": "Rock", "avg_popularity": 29.4, "hit_rate": 6.2 },
      { "genre": "Metal", "avg_popularity": 25.1, "hit_rate": 3.5 },
      { "genre": "Classical", "avg_popularity": 14.2, "hit_rate": 0.8 }
    ],
    "temporal_evolution": [
      { "decade": "1960", "tracks": 24500, "avg_dance": 0.51, "avg_energy": 0.52, "hit_pct": 2.1 },
      { "decade": "1970", "tracks": 51200, "avg_dance": 0.53, "avg_energy": 0.57, "hit_pct": 2.9 },
      { "decade": "1980", "tracks": 98400, "avg_dance": 0.55, "avg_energy": 0.62, "hit_pct": 3.4 },
      { "decade": "1990", "tracks": 165000, "avg_dance": 0.56, "avg_energy": 0.64, "hit_pct": 4.1 },
      { "decade": "2000", "tracks": 284000, "avg_dance": 0.57, "avg_energy": 0.67, "hit_pct": 5.2 },
      { "decade": "2010", "tracks": 412000, "avg_dance": 0.61, "avg_energy": 0.65, "hit_pct": 6.8 },
      { "decade": "2020", "tracks": 147254, "avg_dance": 0.64, "avg_energy": 0.63, "hit_pct": 8.5 }
    ]
  }
}

@app.get("/api/stats")
def get_stats():
    return JSONResponse(content=REAL_METRICS)

@app.get("/api/search")
def search_songs(q: str = ""):
    query = q.strip().lower()
    if not query:
        hits = [s for s in songs_database if s.get("is_hit") == 1][:8]
        return {"results": hits}

    matches = []
    for s in songs_database:
        title = (s.get("title") or "").lower()
        artist = (s.get("track_artist") or "").lower()
        if query in title or query in artist:
            matches.append(s)
            if len(matches) >= 15:
                break
    return {"results": matches}

@app.post("/api/predict")
async def predict_hit(request: Request):
    try:
        song = await request.json()
        expected_cols = feature_config["expected_columns"]
        skewed = feature_config["skewed_features"]
        thresholds = feature_config["thresholds"]

        # Build feature vector
        row = {c: 0.0 for c in expected_cols}
        for num_f in ['energy', 'valence', 'danceability', 'acousticness', 'instrumentalness',
                      'speechiness', 'liveness', 'loudness', 'tempo', 'duration_ms', 'time_signature',
                      'release_year', 'explicit_flag']:
            if num_f in song and song[num_f] is not None:
                row[num_f] = float(song[num_f])

        k_col = f"key_{song.get('key', 0)}"
        if k_col in expected_cols:
            row[k_col] = 1.0

        m_col = f"mode_{song.get('mode', 1)}"
        if m_col in expected_cols:
            row[m_col] = 1.0

        g_col = f"genre_l1_{song.get('genre_l1', 'Unknown')}"
        if g_col in expected_cols:
            row[g_col] = 1.0

        X_df = pd.DataFrame([row])[expected_cols].astype("float32")

        # Predictions
        p_rf = float(rf_model.predict_proba(X_df)[0, 1]) if rf_model else 0.5
        p_xgb = float(xgb_model.predict_proba(X_df)[0, 1]) if xgb_model else 0.5

        if tf_model and scaler:
            Xs = X_df.copy()
            for c in skewed:
                Xs[c] = np.log1p(Xs[c].clip(lower=0))
            Xs_scaled = scaler.transform(Xs).astype("float32")
            p_mlp = float(tf_model.predict(Xs_scaled, verbose=0)[0, 0])
        else:
            p_mlp = (p_rf + p_xgb) / 2.0

        hit_rf = bool(p_rf >= thresholds["rf"])
        hit_xgb = bool(p_xgb >= thresholds["xgb"])
        hit_mlp = bool(p_mlp >= thresholds["tf"])

        votes = sum([hit_rf, hit_xgb, hit_mlp])
        consensus = "HIT" if votes >= 2 else "NAO-HIT"

        return {
            "random_forest": {
                "probability": round(p_rf, 4),
                "prediction": "HIT" if hit_rf else "NAO-HIT",
                "threshold": thresholds["rf"],
                "confidence": round(abs(p_rf - thresholds["rf"]) * 200, 1)
            },
            "xgboost": {
                "probability": round(p_xgb, 4),
                "prediction": "HIT" if hit_xgb else "NAO-HIT",
                "threshold": thresholds["xgb"],
                "confidence": round(abs(p_xgb - thresholds["xgb"]) * 200, 1)
            },
            "tensorflow": {
                "probability": round(p_mlp, 4),
                "prediction": "HIT" if hit_mlp else "NAO-HIT",
                "threshold": thresholds["tf"],
                "confidence": round(abs(p_mlp - thresholds["tf"]) * 200, 1)
            },
            "consensus": consensus,
            "votes_hit": votes,
            "votes_total": 3
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files to serve the frontend
if os.path.exists(PUBLIC_DIR):
    app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="static")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
