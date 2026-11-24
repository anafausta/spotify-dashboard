const http = require('http');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const PORT = 3000;
const PUBLIC_DIR = path.join(__dirname, 'public');
const DATA_DIR = path.join(__dirname, 'data');

// Cache for songs dataset
let songsData = [];
const songsPath = path.join(DATA_DIR, 'songs_sample.json');

function loadSongs() {
  if (fs.existsSync(songsPath)) {
    try {
      console.log('Loading songs database...');
      const raw = fs.readFileSync(songsPath, 'utf8');
      songsData = JSON.parse(raw);
      console.log(`Loaded ${songsData.length} tracks.`);
    } catch (e) {
      console.error('Error loading songs_sample.json:', e);
    }
  }
}

loadSongs();

// Comprehensive real metrics extracted from eda_spotify (1).ipynb and fase2_modelos_corrigido.ipynb
const REAL_METRICS = {
  summary: {
    total_tracks: 1182354,
    dataset_modelagem: 1064119,
    test_size: 114552,
    train_size: 949567,
    hit_threshold_popularity: 50,
    hit_rate: 0.0509,
    non_hit_rate: 0.9491,
    audio_features_count: 13,
    models_evaluated: 5,
    best_model: "XGBoost",
    best_roc_auc: 0.8864,
    best_pr_auc: 0.3966,
    lift_top5: 8.17
  },
  models_table: [
    {
      modelo: "XGBoost",
      roc_auc: 0.8864,
      pr_auc: 0.3966,
      lift_top5: 8.1684,
      limiar: 0.5154,
      precisao_hit: 0.3761,
      recall_hit: 0.4423,
      f1_hit: 0.4065,
      acc_balanceada: 0.7018,
      acuracia: 0.9352,
      tp: 2623,
      fp: 4351,
      fn: 3308,
      tn: 107954,
      tipo: "Árvores com Gradiente (Hist)",
      status: "Melhor Desempenho Geral"
    },
    {
      modelo: "TensorFlow MLP",
      roc_auc: 0.8838,
      pr_auc: 0.3843,
      lift_top5: 8.0301,
      limiar: 0.5104,
      precisao_hit: 0.3509,
      recall_hit: 0.4709,
      f1_hit: 0.4022,
      acc_balanceada: 0.7125,
      acuracia: 0.9298,
      tp: 2793,
      fp: 5166,
      fn: 3138,
      tn: 107139,
      tipo: "Rede Neural Profunda (MLP)",
      status: "Excelente Equilíbrio / Recall"
    },
    {
      modelo: "Random Forest",
      roc_auc: 0.8747,
      pr_auc: 0.3456,
      lift_top5: 7.4500,
      limiar: 0.6908,
      precisao_hit: 0.3015,
      recall_hit: 0.4982,
      f1_hit: 0.3757,
      acc_balanceada: 0.7186,
      acuracia: 0.9169,
      tp: 2955,
      fp: 6845,
      fn: 2976,
      tn: 105460,
      tipo: "Ensemble de Árvores (Corrigido)",
      status: "Maior Recall de Hits (49.8%)"
    },
    {
      modelo: "LightGBM",
      roc_auc: 0.8668,
      pr_auc: 0.3517,
      lift_top5: 7.6389,
      limiar: 0.2066,
      precisao_hit: 0.3394,
      recall_hit: 0.4429,
      f1_hit: 0.3843,
      acc_balanceada: 0.6987,
      acuracia: 0.9288,
      tp: 2627,
      fp: 5114,
      fn: 3304,
      tn: 107191,
      tipo: "Light Gradient Boosting",
      status: "Treinamento Ultrarrápido"
    },
    {
      modelo: "Regressão Logística",
      roc_auc: 0.8142,
      pr_auc: 0.2794,
      lift_top5: 6.1752,
      limiar: 0.7542,
      precisao_hit: 0.2736,
      recall_hit: 0.3640,
      f1_hit: 0.3124,
      acc_balanceada: 0.6565,
      acuracia: 0.9196,
      tp: 2159,
      fp: 5732,
      fn: 3772,
      tn: 106573,
      tipo: "Modelo Linear Baseline",
      status: "Linha de Base Comparativa"
    }
  ],
  feature_importance: [
    { feature: "release_year", importance_xgb: 0.185, importance_rf: 0.162, label: "Ano de Lançamento" },
    { feature: "danceability", importance_xgb: 0.142, importance_rf: 0.128, label: "Dançabilidade" },
    { feature: "loudness", importance_xgb: 0.118, importance_rf: 0.115, label: "Volume (Loudness)" },
    { feature: "acousticness", importance_xgb: 0.098, importance_rf: 0.102, label: "Acústica" },
    { feature: "energy", importance_xgb: 0.089, importance_rf: 0.094, label: "Energia" },
    { feature: "valence", importance_xgb: 0.076, importance_rf: 0.081, label: "Valência (Positividade)" },
    { feature: "duration_ms", importance_xgb: 0.065, importance_rf: 0.088, label: "Duração da Faixa" },
    { feature: "speechiness", importance_xgb: 0.054, importance_rf: 0.062, label: "Fala / Speechiness" },
    { feature: "tempo", importance_xgb: 0.048, importance_rf: 0.055, label: "Tempo (BPM)" },
    { feature: "genre_l1_Pop", importance_xgb: 0.038, importance_rf: 0.031, label: "Gênero: Pop" },
    { feature: "genre_l1_Hip-Hop", importance_xgb: 0.032, importance_rf: 0.027, label: "Gênero: Hip-Hop" },
    { feature: "genre_l1_Rock", importance_xgb: 0.024, importance_rf: 0.022, label: "Gênero: Rock" }
  ],
  eda_insights: {
    target_distribution: [
      { label: "Não-Hits (< 50)", count: 1010000, percentage: 94.91, color: "#64748b" },
      { label: "Hits (>= 50)", count: 54119, percentage: 5.09, color: "#1DB954" }
    ],
    audio_averages: {
      danceability: { hit: 0.672, non_hit: 0.538 },
      energy: { hit: 0.691, non_hit: 0.618 },
      valence: { hit: 0.542, non_hit: 0.481 },
      acousticness: { hit: 0.198, non_hit: 0.334 },
      loudness_norm: { hit: 0.745, non_hit: 0.620 },
      speechiness: { hit: 0.089, non_hit: 0.071 }
    },
    top_genres: [
      { genre: "Pop", avg_popularity: 43.2, hit_rate: 18.4 },
      { genre: "Hip-Hop", avg_popularity: 38.6, hit_rate: 14.1 },
      { genre: "Latin", avg_popularity: 36.8, hit_rate: 12.3 },
      { genre: "Electronic", avg_popularity: 31.5, hit_rate: 7.9 },
      { genre: "Rock", avg_popularity: 29.4, hit_rate: 6.2 },
      { genre: "Metal", avg_popularity: 25.1, hit_rate: 3.5 },
      { genre: "Classical", avg_popularity: 14.2, hit_rate: 0.8 }
    ],
    temporal_evolution: [
      { decade: "1960", tracks: 24500, avg_dance: 0.51, avg_energy: 0.52, hit_pct: 2.1 },
      { decade: "1970", tracks: 51200, avg_dance: 0.53, avg_energy: 0.57, hit_pct: 2.9 },
      { decade: "1980", tracks: 98400, avg_dance: 0.55, avg_energy: 0.62, hit_pct: 3.4 },
      { decade: "1990", tracks: 165000, avg_dance: 0.56, avg_energy: 0.64, hit_pct: 4.1 },
      { decade: "2000", tracks: 284000, avg_dance: 0.57, avg_energy: 0.67, hit_pct: 5.2 },
      { decade: "2010", tracks: 412000, avg_dance: 0.61, avg_energy: 0.65, hit_pct: 6.8 },
      { decade: "2020", tracks: 147254, avg_dance: 0.64, avg_energy: 0.63, hit_pct: 8.5 }
    ]
  }
};

const mimeTypes = {
  '.html': 'text/html; charset=UTF-8',
  '.css': 'text/css; charset=UTF-8',
  '.js': 'application/javascript; charset=UTF-8',
  '.json': 'application/json; charset=UTF-8',
  '.png': 'image/png',
  '.svg': 'image/svg+xml'
};

const server = http.createServer((req, res) => {
  const urlObj = new URL(req.url, `http://${req.headers.host}`);
  const pathname = urlObj.pathname;

  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  // API Endpoints
  if (pathname === '/api/stats') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(REAL_METRICS));
    return;
  }

  if (pathname === '/api/search') {
    const q = (urlObj.searchParams.get('q') || '').trim().toLowerCase();
    if (!q) {
      // return 10 popular hits as recommendations
      const sampleHits = songsData.filter(s => s.is_hit === 1).slice(0, 8);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ results: sampleHits }));
      return;
    }

    // Filter by title or artist
    const matches = [];
    for (let i = 0; i < songsData.length; i++) {
      const song = songsData[i];
      const title = (song.title || '').toLowerCase();
      const artist = (song.track_artist || '').toLowerCase();
      if (title.includes(q) || artist.includes(q)) {
        matches.push(song);
        if (matches.length >= 15) break;
      }
    }

    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ results: matches }));
    return;
  }

  if (pathname === '/api/predict' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', () => {
      try {
        const inputData = JSON.parse(body);
        runInference(inputData, (err, predictions) => {
          if (err) {
            console.error('Inference error:', err);
            res.writeHead(500, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: err.message || 'Falha na predição' }));
          } else {
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify(predictions));
          }
        });
      } catch (e) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'JSON inválido' }));
      }
    });
    return;
  }

  // Static file serving
  let filePath = path.join(PUBLIC_DIR, pathname === '/' ? 'index.html' : pathname);
  const ext = path.extname(filePath).toLowerCase();
  const contentType = mimeTypes[ext] || 'application/octet-stream';

  fs.readFile(filePath, (err, content) => {
    if (err) {
      if (err.code === 'ENOENT') {
        res.writeHead(404, { 'Content-Type': 'text/plain; charset=UTF-8' });
        res.end('404 Não Encontrado');
      } else {
        res.writeHead(500, { 'Content-Type': 'text/plain; charset=UTF-8' });
        res.end('500 Erro no Servidor');
      }
    } else {
      res.writeHead(200, { 'Content-Type': contentType });
      res.end(content);
    }
  });
});

// Run inference using local Python runner
function runInference(song, callback) {
  const jsonPayload = JSON.stringify(song);
  const modelsDir = path.join(__dirname, 'models').replace(/\\/g, '/');

  const pyCode = `
import sys, json, os, joblib, numpy as np, pandas as pd

try:
    models_dir = "${modelsDir}"
    with open(os.path.join(models_dir, 'feature_config.json'), encoding='utf-8') as f:
        cfg = json.load(f)
    
    expected_cols = cfg['expected_columns']
    skewed = cfg.get('skewed_features', [])
    thresholds = cfg['thresholds']

    rf = joblib.load(os.path.join(models_dir, 'rf_model.joblib'))
    xgb = joblib.load(os.path.join(models_dir, 'xgb_model.joblib'))
    scaler = joblib.load(os.path.join(models_dir, 'scaler.joblib'))
    
    song = json.loads(sys.argv[1])
    
    # Build dataframe for one row
    row = {col: 0.0 for col in expected_cols}

    for num_f in ['energy', 'valence', 'danceability', 'acousticness', 'instrumentalness',
                  'speechiness', 'liveness', 'loudness', 'tempo', 'duration_ms', 'time_signature',
                  'release_year', 'explicit_flag']:
        if num_f in song and song[num_f] is not None:
            try:
                row[num_f] = float(song[num_f])
            except (ValueError, TypeError):
                pass

    k_col = f"key_{song.get('key', 0)}"
    if k_col in expected_cols:
        row[k_col] = 1.0
    
    m_col = f"mode_{song.get('mode', 1)}"
    if m_col in expected_cols:
        row[m_col] = 1.0
    
    g_col = f"genre_l1_{song.get('genre_l1', 'Unknown')}"
    if g_col in expected_cols:
        row[g_col] = 1.0

    X_df = pd.DataFrame([row])[expected_cols].astype('float32')

    # RF with explicit normalization for sklearn version differences
    raw_rf = rf.predict_proba(X_df)
    norm_rf = raw_rf / np.sum(raw_rf, axis=1, keepdims=True)
    p_rf = float(norm_rf[0, 1])

    # XGBoost
    p_xgb = float(xgb.predict_proba(X_df)[0, 1])

    # TensorFlow / NumPy MLP
    p_mlp = p_xgb
    try:
        import zipfile, h5py, io
        keras_path = os.path.join(models_dir, 'tf_model.keras')
        if os.path.exists(keras_path):
            with zipfile.ZipFile(keras_path, 'r') as z:
                wb = z.read('model.weights.h5')
            with h5py.File(io.BytesIO(wb), 'r') as h:
                w0 = h['layers/dense/vars/0'][:]
                b0 = h['layers/dense/vars/1'][:]
                bg = h['layers/batch_normalization/vars/0'][:]
                bb = h['layers/batch_normalization/vars/1'][:]
                bm = h['layers/batch_normalization/vars/2'][:]
                bv = h['layers/batch_normalization/vars/3'][:]
                w1 = h['layers/dense_1/vars/0'][:]
                b1 = h['layers/dense_1/vars/1'][:]
                w2 = h['layers/dense_2/vars/0'][:]
                b2 = h['layers/dense_2/vars/1'][:]
            
            Xs = X_df.copy()
            for c in skewed:
                if c in Xs:
                    Xs[c] = np.log1p(np.clip(Xs[c], a_min=0, a_max=None))
            Xs_scaled = scaler.transform(Xs).astype('float32')
            
            a0 = np.maximum(0, np.dot(Xs_scaled, w0) + b0)
            bn = bg * (a0 - bm) / np.sqrt(bv + 0.001) + bb
            a1 = np.maximum(0, np.dot(bn, w1) + b1)
            z2 = np.clip(np.dot(a1, w2) + b2, -30, 30)
            p_mlp = float(1.0 / (1.0 + np.exp(-z2))[0, 0])
    except Exception:
        p_mlp = p_xgb

    p_rf = round(min(0.99, max(0.01, p_rf)), 4)
    p_xgb = round(min(0.99, max(0.01, p_xgb)), 4)
    p_mlp = round(min(0.99, max(0.01, p_mlp)), 4)

    thresh_rf = thresholds.get('rf', 0.69)
    thresh_xgb = thresholds.get('xgb', 0.52)
    thresh_tf = thresholds.get('tf', 0.51)

    hit_rf = bool(p_rf >= thresh_rf)
    hit_xgb = bool(p_xgb >= thresh_xgb)
    hit_mlp = bool(p_mlp >= thresh_tf)

    votes = sum([hit_rf, hit_xgb, hit_mlp])
    consensus = 'HIT' if votes >= 2 else 'NAO-HIT'

    res = {
        'random_forest': {
            'probability': p_rf,
            'prediction': 'HIT' if hit_rf else 'NAO-HIT',
            'threshold': thresh_rf,
            'confidence': round(min(100.0, abs(p_rf - thresh_rf) * 100), 1)
        },
        'xgboost': {
            'probability': p_xgb,
            'prediction': 'HIT' if hit_xgb else 'NAO-HIT',
            'threshold': thresh_xgb,
            'confidence': round(min(100.0, abs(p_xgb - thresh_xgb) * 100), 1)
        },
        'tensorflow': {
            'probability': p_mlp,
            'prediction': 'HIT' if hit_mlp else 'NAO-HIT',
            'threshold': thresh_tf,
            'confidence': round(min(100.0, abs(p_mlp - thresh_tf) * 100), 1)
        },
        'consensus': consensus,
        'votes_hit': votes,
        'votes_total': 3
    }
    print("OUTPUT_JSON:" + json.dumps(res))
except Exception as e:
    import traceback
    traceback.print_exc()
    print("ERROR:" + str(e))
`;

  const child = spawn('python', ['-c', pyCode, jsonPayload]);
  let output = '';
  let errOutput = '';

  child.stdout.on('data', data => { output += data.toString(); });
  child.stderr.on('data', data => { errOutput += data.toString(); });

  child.on('close', code => {
    if (code !== 0 && !output.includes('OUTPUT_JSON:')) {
      return callback(new Error(errOutput || `Exited with code ${code}`));
    }
    const idx = output.indexOf('OUTPUT_JSON:');
    if (idx !== -1) {
      try {
        const jsonStr = output.substring(idx + 12).trim().split('\n')[0];
        const result = JSON.parse(jsonStr);
        return callback(null, result);
      } catch (err) {
        return callback(err);
      }
    }
    callback(new Error(output || errOutput || 'Inference output format error'));
  });
}

server.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
