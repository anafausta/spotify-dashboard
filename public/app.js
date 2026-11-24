// Spotify Hit Prediction & Analytics App

let statsData = null;
let currentSelectedSong = null;
let searchDebounceTimeout = null;
let charts = {};

document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initTabs();
  loadStats();
  initPredictSearch();
});

// ================= THEME TOGGLE (LIGHT / DARK) =================
function initTheme() {
  const toggleBtn = document.getElementById('theme-toggle-btn');
  const themeIcon = document.getElementById('theme-icon');
  const themeText = document.getElementById('theme-text');

  // Padrão é Modo Claro. Se o usuário já escolheu escuro anteriormente, recupera:
  const savedTheme = localStorage.getItem('spotify_theme') || 'light';
  if (savedTheme === 'dark') {
    document.body.classList.add('dark-theme');
    if (themeIcon) themeIcon.textContent = '☀️';
    if (themeText) themeText.textContent = 'Modo Claro';
  } else {
    document.body.classList.remove('dark-theme');
    if (themeIcon) themeIcon.textContent = '🌙';
    if (themeText) themeText.textContent = 'Modo Escuro';
  }

  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      const isDark = document.body.classList.toggle('dark-theme');
      localStorage.setItem('spotify_theme', isDark ? 'dark' : 'light');

      if (themeIcon) themeIcon.textContent = isDark ? '☀️' : '🌙';
      if (themeText) themeText.textContent = isDark ? 'Modo Claro' : 'Modo Escuro';

      // Atualiza os gráficos para as cores do novo tema
      if (statsData) {
        destroyCharts();
        initCharts(statsData);
      }
    });
  }
}

function destroyCharts() {
  Object.values(charts).forEach(ch => {
    if (ch && typeof ch.destroy === 'function') {
      ch.destroy();
    }
  });
  charts = {};
}

// ================= TAB SWITCHING =================
function initTabs() {
  const tabButtons = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');

  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTabId = btn.getAttribute('data-tab');
      tabButtons.forEach(b => b.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));

      btn.classList.add('active');
      const activeContent = document.getElementById(targetTabId);
      if (activeContent) {
        activeContent.classList.add('active');
      }

      // Redesenha gráficos se necessário
      if (targetTabId === 'tab-dashboard') {
        Object.values(charts).forEach(ch => ch.resize && ch.resize());
      }
    });
  });
}

// ================= LOAD STATS FROM BACKEND =================
async function loadStats() {
  try {
    const res = await fetch('/api/stats');
    statsData = await res.json();
    renderSummaryKPIs(statsData.summary);
    renderModelsTable(statsData.models_table);
    renderMatrixPills(statsData.models_table);
    initCharts(statsData);
  } catch (err) {
    console.error('Falha ao carregar estatísticas:', err);
  }
}

// Render Summary KPIs
function renderSummaryKPIs(summary) {
  if (!summary) return;
  document.getElementById('kpi-tracks').textContent = Number(summary.total_tracks).toLocaleString('pt-BR');
  document.getElementById('kpi-hitrate').textContent = (summary.hit_rate * 100).toFixed(2) + '%';
  document.getElementById('kpi-best-roc').textContent = summary.best_roc_auc.toFixed(4);
  document.getElementById('kpi-lift').textContent = summary.lift_top5.toFixed(2) + 'x';
}

// Render Models Comparison Table
function renderModelsTable(models) {
  const tbody = document.getElementById('models-table-body');
  if (!tbody || !models) return;

  tbody.innerHTML = models.map((m, idx) => `
    <tr>
      <td>
        <div class="table-model-name">
          ${escapeHtml(m.modelo)}
          ${idx === 0 ? '<span class="best-badge">Melhor Modelo</span>' : ''}
        </div>
      </td>
      <td class="metric-highlight">${m.roc_auc.toFixed(4)}</td>
      <td class="metric-highlight" style="color: var(--accent-cyan); font-weight: 750;">${m.pr_auc.toFixed(4)}</td>
      <td>${m.f1_hit.toFixed(4)}</td>
      <td>${(m.recall_hit * 100).toFixed(1)}%</td>
      <td>${(m.precisao_hit * 100).toFixed(1)}%</td>
      <td>${(m.acuracia * 100).toFixed(2)}%</td>
      <td><span class="badge">${m.limiar.toFixed(2)}</span></td>
      <td style="color: var(--accent-cyan); font-weight: 750;">${m.lift_top5.toFixed(2)}x</td>
      <td style="color: var(--text-secondary); font-size: 0.8rem; font-weight: 600;">${escapeHtml(m.status)}</td>
    </tr>
  `).join('');
}

// Render Confusion Matrix Pill Selector
function renderMatrixPills(models) {
  const container = document.getElementById('matrix-model-pills');
  if (!container || !models) return;

  container.innerHTML = models.map((m, idx) => `
    <button class="matrix-pill ${idx === 0 ? 'active' : ''}" data-index="${idx}">
      ${escapeHtml(m.modelo)}
    </button>
  `).join('');

  container.querySelectorAll('.matrix-pill').forEach(btn => {
    btn.addEventListener('click', () => {
      container.querySelectorAll('.matrix-pill').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const idx = parseInt(btn.getAttribute('data-index'), 10);
      renderConfusionMatrix(models[idx]);
    });
  });

  if (models.length > 0) {
    renderConfusionMatrix(models[0]);
  }
}

// Render Confusion Matrix Quadrants
function renderConfusionMatrix(m) {
  const container = document.getElementById('matrix-display-container');
  if (!container || !m) return;

  container.innerHTML = `
    <div class="matrix-quadrant tp">
      <div class="quad-name">Verdadeiro Positivo (TP)</div>
      <div class="quad-val" style="color: var(--spotify-green);">${Number(m.tp).toLocaleString('pt-BR')}</div>
      <div class="quad-desc">Hits reais corretamente identificados</div>
    </div>
    <div class="matrix-quadrant fp">
      <div class="quad-name">Falso Positivo (FP)</div>
      <div class="quad-val" style="color: var(--accent-amber);">${Number(m.fp).toLocaleString('pt-BR')}</div>
      <div class="quad-desc">Não-hits previstos como hit (alarme falso)</div>
    </div>
    <div class="matrix-quadrant fn">
      <div class="quad-name">Falso Negativo (FN)</div>
      <div class="quad-val" style="color: var(--accent-rose);">${Number(m.fn).toLocaleString('pt-BR')}</div>
      <div class="quad-desc">Hits reais que o modelo deixou passar</div>
    </div>
    <div class="matrix-quadrant tn">
      <div class="quad-name">Verdadeiro Negativo (TN)</div>
      <div class="quad-val" style="color: var(--accent-blue);">${Number(m.tn).toLocaleString('pt-BR')}</div>
      <div class="quad-desc">Não-hits descartados corretamente</div>
    </div>
  `;
}

// Initialize Interactive Charts
function initCharts(data) {
  const isDark = document.body.classList.contains('dark-theme');
  const textColor = isDark ? '#a1a1aa' : '#334155';
  const textMuted = isDark ? '#71717a' : '#64748b';
  const gridColor = isDark ? 'rgba(255, 255, 255, 0.06)' : 'rgba(0, 0, 0, 0.06)';

  Chart.defaults.color = textColor;
  Chart.defaults.font.family = "'Plus Jakarta Sans', system-ui, sans-serif";
  Chart.defaults.plugins.tooltip.backgroundColor = isDark ? '#1e1e24' : '#0f172a';
  Chart.defaults.plugins.tooltip.titleColor = '#ffffff';
  Chart.defaults.plugins.tooltip.bodyColor = '#f1f5f9';
  Chart.defaults.plugins.tooltip.borderColor = isDark ? '#3f3f46' : '#cbd5e1';
  Chart.defaults.plugins.tooltip.borderWidth = 1;
  Chart.defaults.plugins.tooltip.padding = 10;
  Chart.defaults.plugins.tooltip.cornerRadius = 8;

  // Chart 1: Models Comparison (ROC-AUC vs PR-AUC)
  const ctxModels = document.getElementById('chartModelsComparison');
  if (ctxModels && data.models_table) {
    charts.models = new Chart(ctxModels, {
      type: 'bar',
      data: {
        labels: data.models_table.map(m => m.modelo),
        datasets: [
          {
            label: 'ROC-AUC',
            data: data.models_table.map(m => m.roc_auc),
            backgroundColor: '#15803d',
            borderRadius: 6
          },
          {
            label: 'PR-AUC (Foco Desbalanceamento)',
            data: data.models_table.map(m => m.pr_auc),
            backgroundColor: '#0284c7',
            borderRadius: 6
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'top', labels: { boxWidth: 14, font: { weight: 600 } } }
        },
        scales: {
          y: { min: 0, max: 1.0, grid: { color: gridColor }, ticks: { color: textMuted } },
          x: { grid: { display: false }, ticks: { color: textColor, font: { weight: 600 } } }
        }
      }
    });
  }

  // Chart 2: Feature Importance
  const ctxFeat = document.getElementById('chartFeatureImportance');
  if (ctxFeat && data.feature_importance) {
    const topFeats = data.feature_importance.slice(0, 8);
    charts.features = new Chart(ctxFeat, {
      type: 'bar',
      data: {
        labels: topFeats.map(f => f.label),
        datasets: [
          {
            label: 'XGBoost',
            data: topFeats.map(f => f.importance_xgb),
            backgroundColor: '#15803d',
            borderRadius: 6
          },
          {
            label: 'Random Forest',
            data: topFeats.map(f => f.importance_rf),
            backgroundColor: '#2563eb',
            borderRadius: 6
          }
        ]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'top', labels: { boxWidth: 14, font: { weight: 600 } } }
        },
        scales: {
          x: { grid: { color: gridColor }, ticks: { color: textMuted } },
          y: { grid: { display: false }, ticks: { color: textColor, font: { weight: 600 } } }
        }
      }
    });
  }

  // Chart 3: Target Distribution Doughnut
  const ctxTarget = document.getElementById('chartTargetDistribution');
  if (ctxTarget && data.eda_insights) {
    charts.target = new Chart(ctxTarget, {
      type: 'doughnut',
      data: {
        labels: data.eda_insights.target_distribution.map(t => t.label),
        datasets: [{
          data: data.eda_insights.target_distribution.map(t => t.count),
          backgroundColor: ['#94a3b8', '#15803d'],
          borderWidth: 2,
          borderColor: isDark ? '#1e1e24' : '#ffffff'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'bottom', labels: { boxWidth: 14, font: { weight: 600 } } }
        },
        cutout: '70%'
      }
    });
  }

  // Chart 4: Audio Profile
  const ctxAudio = document.getElementById('chartAudioProfile');
  if (ctxAudio && data.eda_insights) {
    const p = data.eda_insights.audio_averages;
    charts.audio = new Chart(ctxAudio, {
      type: 'bar',
      data: {
        labels: ['Dançabilidade', 'Energia', 'Valência', 'Acústica', 'Speechiness'],
        datasets: [
          {
            label: 'Hits (≥ 50)',
            data: [p.danceability.hit, p.energy.hit, p.valence.hit, p.acousticness.hit, p.speechiness.hit],
            backgroundColor: '#15803d',
            borderRadius: 6
          },
          {
            label: 'Não-Hits',
            data: [p.danceability.non_hit, p.energy.non_hit, p.valence.non_hit, p.acousticness.non_hit, p.speechiness.non_hit],
            backgroundColor: '#94a3b8',
            borderRadius: 6
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'top', labels: { boxWidth: 14, font: { weight: 600 } } }
        },
        scales: {
          y: { min: 0, max: 0.8, grid: { color: gridColor }, ticks: { color: textMuted } },
          x: { grid: { display: false }, ticks: { color: textColor, font: { weight: 600 } } }
        }
      }
    });
  }

  // Chart 5: Genres Performance
  const ctxGenres = document.getElementById('chartGenres');
  if (ctxGenres && data.eda_insights) {
    charts.genres = new Chart(ctxGenres, {
      type: 'bar',
      data: {
        labels: data.eda_insights.top_genres.map(g => g.genre),
        datasets: [{
          label: '% de Hits no Gênero',
          data: data.eda_insights.top_genres.map(g => g.hit_rate),
          backgroundColor: '#0284c7',
          borderRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false }
        },
        scales: {
          y: {
            grid: { color: gridColor },
            ticks: { callback: v => v + '%', color: textMuted }
          },
          x: { grid: { display: false }, ticks: { color: textColor, font: { weight: 600 } } }
        }
      }
    });
  }

  // Chart 6: Temporal Evolution
  const ctxTemporal = document.getElementById('chartTemporal');
  if (ctxTemporal && data.eda_insights) {
    const t = data.eda_insights.temporal_evolution;
    charts.temporal = new Chart(ctxTemporal, {
      type: 'line',
      data: {
        labels: t.map(d => 'Década ' + d.decade),
        datasets: [
          {
            label: 'Dançabilidade Média',
            data: t.map(d => d.avg_dance),
            borderColor: '#15803d',
            backgroundColor: 'rgba(21, 128, 61, 0.1)',
            fill: true,
            tension: 0.35,
            borderWidth: 2.5
          },
          {
            label: 'Energia Média',
            data: t.map(d => d.avg_energy),
            borderColor: '#b45309',
            backgroundColor: 'transparent',
            tension: 0.35,
            borderWidth: 2.5
          },
          {
            label: 'Taxa de Hits (%) / 10',
            data: t.map(d => d.hit_pct / 10),
            borderColor: '#0284c7',
            borderDash: [5, 5],
            tension: 0.35,
            borderWidth: 2.5
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'top', labels: { boxWidth: 14, font: { weight: 600 } } }
        },
        scales: {
          y: { grid: { color: gridColor }, ticks: { color: textMuted } },
          x: { grid: { display: false }, ticks: { color: textColor, font: { weight: 600 } } }
        }
      }
    });
  }
}

// ================= PREDICT / SEARCH LOGIC =================
function initPredictSearch() {
  const input = document.getElementById('song-search-input');
  const clearBtn = document.getElementById('clear-search-btn');
  const quickPills = document.querySelectorAll('.quick-pill');
  const reRunBtn = document.getElementById('re-run-btn');

  // Carrega sugestões populares iniciais
  fetchResults('');

  input.addEventListener('input', (e) => {
    const q = e.target.value.trim();
    clearBtn.style.display = q ? 'block' : 'none';

    clearTimeout(searchDebounceTimeout);
    searchDebounceTimeout = setTimeout(() => {
      fetchResults(q);
    }, 250);
  });

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      const q = e.target.value.trim();
      if (q) {
        clearTimeout(searchDebounceTimeout);
        fetchResults(q, true);
      }
    }
  });

  clearBtn.addEventListener('click', () => {
    input.value = '';
    clearBtn.style.display = 'none';
    fetchResults('');
    input.focus();
  });

  quickPills.forEach(pill => {
    pill.addEventListener('click', () => {
      const songTitle = pill.getAttribute('data-song');
      input.value = songTitle;
      clearBtn.style.display = 'block';
      fetchResults(songTitle, true);
    });
  });

  if (reRunBtn) {
    reRunBtn.addEventListener('click', () => {
      if (currentSelectedSong) {
        executeModelsInference(currentSelectedSong);
      }
    });
  }
}

async function fetchResults(q, autoSelectFirst = false) {
  const container = document.getElementById('search-results-container');
  try {
    const res = await fetch(`/api/search?q=${encodeURIComponent(q)}`);
    const data = await res.json();
    renderSearchResults(data.results || [], autoSelectFirst);
  } catch (err) {
    console.error('Erro na busca de músicas:', err);
    container.innerHTML = `<div class="empty-state" style="padding: 1.5rem;"><p>Erro ao conectar ao banco de dados de músicas.</p></div>`;
  }
}

function renderSearchResults(songs, autoSelectFirst = false) {
  const container = document.getElementById('search-results-container');
  if (!container) return;

  if (songs.length === 0) {
    container.innerHTML = `
      <div class="empty-state" style="padding: 1.5rem;">
        <p>Nenhuma música correspondente encontrada no catálogo.</p>
      </div>
    `;
    return;
  }

  container.innerHTML = songs.map(song => `
    <div class="search-result-item" data-id="${escapeHtml(song.title)}">
      <div class="result-main">
        <div class="result-title">${escapeHtml(song.title)}</div>
        <div class="result-sub">${escapeHtml(song.track_artist || 'Desconhecido')} &bull; ${song.release_year || 'N/A'} &bull; ${escapeHtml(song.genre_l1 || 'Outro')}</div>
      </div>
      <div>
        <span class="result-tag ${song.is_hit === 1 ? 'tag-hit' : 'tag-nohit'}">
          ${song.ai_synthesized ? '✨ IA ' : ''}${song.is_hit === 1 ? 'Hit' : 'Não-Hit'}
        </span>
      </div>
    </div>
  `).join('');

  const items = container.querySelectorAll('.search-result-item');
  items.forEach((item, idx) => {
    item.addEventListener('click', () => {
      items.forEach(it => it.classList.remove('active'));
      item.classList.add('active');
      selectSong(songs[idx]);
    });
  });

  if (autoSelectFirst && songs.length > 0) {
    items[0].classList.add('active');
    selectSong(songs[0]);
  }
}

function selectSong(song) {
  currentSelectedSong = song;

  // Mostra painel de detalhes e esconde estado vazio
  document.getElementById('predict-empty-state').style.display = 'none';
  const card = document.getElementById('predict-active-card');
  card.style.display = 'block';

  // Metadados da Música
  document.getElementById('meta-title').textContent = song.title;
  document.getElementById('meta-artist').textContent = song.track_artist || 'Artista Desconhecido';
  document.getElementById('meta-genre').textContent = song.genre_l1 || 'Gênero Indefinido';
  document.getElementById('meta-year').textContent = song.release_year || 'Ano N/A';

  const ms = song.duration_ms || 200000;
  const mins = Math.floor(ms / 60000);
  const secs = Math.floor((ms % 60000) / 1000).toString().padStart(2, '0');
  document.getElementById('meta-duration').textContent = `${mins}:${secs}`;
  document.getElementById('meta-key').textContent = `Tom: ${getKeyName(song.key, song.mode)}`;
  document.getElementById('meta-explicit').textContent = song.explicit_flag ? 'Explícita (+18)' : 'Classificação Livre';

  // Veredito Real do Dataset
  const pop = song.track_popularity || 0;
  const isHit = song.is_hit === 1 || pop >= 50;

  document.getElementById('meta-popularity-val').textContent = `${pop}/100`;
  document.getElementById('meta-popularity-bar').style.width = `${pop}%`;

  const statusBadge = document.getElementById('dataset-status-badge');
  if (isHit) {
    statusBadge.textContent = 'HIT 🏆';
    statusBadge.className = 'verdict-status verdict-hit';
  } else {
    statusBadge.textContent = 'NÃO-HIT 🎵';
    statusBadge.className = 'verdict-status verdict-nohit';
  }

  // Características acústicas
  const dance = (song.danceability || 0).toFixed(2);
  const energy = (song.energy || 0).toFixed(2);
  const valence = (song.valence || 0).toFixed(2);
  const acoustic = (song.acousticness || 0).toFixed(2);
  const loudness = song.loudness != null ? song.loudness.toFixed(1) + ' dB' : '-8.0 dB';
  const tempo = song.tempo ? Math.round(song.tempo) + ' BPM' : '120 BPM';

  document.getElementById('feat-dance').textContent = dance;
  document.getElementById('bar-dance').style.width = `${Math.min(100, (song.danceability || 0) * 100)}%`;

  document.getElementById('feat-energy').textContent = energy;
  document.getElementById('bar-energy').style.width = `${Math.min(100, (song.energy || 0) * 100)}%`;

  document.getElementById('feat-valence').textContent = valence;
  document.getElementById('bar-valence').style.width = `${Math.min(100, (song.valence || 0) * 100)}%`;

  document.getElementById('feat-acoustic').textContent = acoustic;
  document.getElementById('bar-acoustic').style.width = `${Math.min(100, (song.acousticness || 0) * 100)}%`;

  document.getElementById('feat-loudness').textContent = loudness;
  document.getElementById('bar-loudness').style.width = `${Math.max(10, Math.min(100, ((song.loudness || -20) + 30) * 3))}%`;

  document.getElementById('feat-tempo').textContent = tempo;
  document.getElementById('bar-tempo').style.width = `${Math.min(100, ((song.tempo || 120) / 200) * 100)}%`;

  // Reset visual state dos modelos enquanto calcula
  document.getElementById('rf-status').textContent = 'Calculando...';
  document.getElementById('rf-status').className = 'model-pred-status';
  document.getElementById('rf-prob').textContent = '--%';
  document.getElementById('rf-bar').style.width = '20%';

  document.getElementById('xgb-status').textContent = 'Calculando...';
  document.getElementById('xgb-status').className = 'model-pred-status';
  document.getElementById('xgb-prob').textContent = '--%';
  document.getElementById('xgb-bar').style.width = '20%';

  document.getElementById('tf-status').textContent = 'Calculando...';
  document.getElementById('tf-status').className = 'model-pred-status';
  document.getElementById('tf-prob').textContent = '--%';
  document.getElementById('tf-bar').style.width = '20%';

  document.getElementById('ai-user-badge').textContent = '⏳ Analisando...';
  document.getElementById('ai-summary-text').textContent = 'Analisando características musicais com inteligência artificial...';
  document.getElementById('ai-user-desc-text').textContent = 'Carregando comportamento e recepção nas plataformas de streaming...';
  document.getElementById('ai-model-rationale-text').textContent = 'Avaliando correlação de dançabilidade, loudness e volume com os limiares de decisão...';

  // Executa predição em tempo real nos 3 modelos
  executeModelsInference(song);
}

// Call Backend to Run RandomForest, XGBoost, and TensorFlow
async function executeModelsInference(song) {
  const loading = document.getElementById('inference-loading');
  loading.style.display = 'flex';

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(song)
    });

    const data = await res.json();
    loading.style.display = 'none';

    renderInferenceResults(data);
  } catch (err) {
    loading.style.display = 'none';
    console.error('Erro na chamada de inferência:', err);
  }
}

function renderInferenceResults(pred) {
  const fmt = (p) => p === 'HIT' ? 'HIT' : 'NÃO-HIT';

  // 1. Random Forest
  const rf = pred.random_forest;
  const isRfHit = rf.prediction === 'HIT';
  const rfStatus = document.getElementById('rf-status');
  rfStatus.textContent = fmt(rf.prediction);
  rfStatus.className = `model-pred-status ${isRfHit ? 'status-hit' : 'status-nohit'}`;
  document.getElementById('rf-prob').textContent = (rf.probability * 100).toFixed(1) + '%';
  const rfBar = document.getElementById('rf-bar');
  rfBar.style.width = `${Math.max(5, rf.probability * 100)}%`;
  rfBar.className = `prob-bar-fill ${isRfHit ? '' : 'bar-nohit'}`;

  // 2. XGBoost
  const xgb = pred.xgboost;
  const isXgbHit = xgb.prediction === 'HIT';
  const xgbStatus = document.getElementById('xgb-status');
  xgbStatus.textContent = fmt(xgb.prediction);
  xgbStatus.className = `model-pred-status ${isXgbHit ? 'status-hit' : 'status-nohit'}`;
  document.getElementById('xgb-prob').textContent = (xgb.probability * 100).toFixed(1) + '%';
  const xgbBar = document.getElementById('xgb-bar');
  xgbBar.style.width = `${Math.max(5, xgb.probability * 100)}%`;
  xgbBar.className = `prob-bar-fill ${isXgbHit ? '' : 'bar-nohit'}`;

  // 3. TensorFlow MLP
  const tf = pred.tensorflow;
  const isTfHit = tf.prediction === 'HIT';
  const tfStatus = document.getElementById('tf-status');
  tfStatus.textContent = fmt(tf.prediction);
  tfStatus.className = `model-pred-status ${isTfHit ? 'status-hit' : 'status-nohit'}`;
  document.getElementById('tf-prob').textContent = (tf.probability * 100).toFixed(1) + '%';
  const tfBar = document.getElementById('tf-bar');
  tfBar.style.width = `${Math.max(5, tf.probability * 100)}%`;
  tfBar.className = `prob-bar-fill ${isTfHit ? '' : 'bar-nohit'}`;

  // 4. Painel de Inteligência Artificial
  if (pred.ai_analysis) {
    const ai = pred.ai_analysis;
    const badge = document.getElementById('ai-user-badge');
    badge.textContent = ai.user_classification || '🔥 Fenômeno de Streaming';
    document.getElementById('ai-summary-text').textContent = ai.ai_summary || '';
    document.getElementById('ai-user-desc-text').textContent = ai.user_desc || '';
    document.getElementById('ai-model-rationale-text').textContent = ai.model_rationale || '';
  }

  // 5. Consensus Box
  const consensusBox = document.getElementById('consensus-box');
  const consensusIcon = document.getElementById('consensus-icon');
  const consensusTitle = document.getElementById('consensus-title');
  const consensusDesc = document.getElementById('consensus-desc');

  if (pred.consensus === 'HIT') {
    consensusBox.className = 'consensus-banner';
    consensusIcon.textContent = '🏆';
    consensusTitle.textContent = `Consenso dos Modelos: HIT (${pred.votes_hit} de 3 votos)`;
    consensusDesc.textContent = `Os algoritmos convergiram apontando alta probabilidade comercial com base no perfil sonoro da faixa.`;
  } else {
    consensusBox.className = 'consensus-banner banner-nohit';
    consensusIcon.textContent = '🎵';
    consensusTitle.textContent = `Consenso dos Modelos: NÃO-HIT (${3 - pred.votes_hit} de 3 votos)`;
    consensusDesc.textContent = `As características sonoras da música situam-se fora do padrão típico de sucessos comerciais massivos no Spotify.`;
  }
}

// Helpers
function getKeyName(key, mode) {
  const keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  const k = keys[key] || 'C';
  const m = mode === 1 ? 'Maior' : 'Menor';
  return `${k} ${m}`;
}

function escapeHtml(text) {
  const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
  return String(text).replace(/[&<>"']/g, m => map[m]);
}
