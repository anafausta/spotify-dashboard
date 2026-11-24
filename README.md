---
title: Spotify Intelligence - Hit Prediction & EDA
emoji: 🎵
colorFrom: green
colorTo: emerald
sdk: docker
app_port: 7860
pinned: false
---

# 🎵 Spotify Intelligence — EDA & Machine Learning Hit Prediction

Aplicação completa de Ciência de Dados e Machine Learning para análise exploratória e previsão de sucessos comerciais (Hits) na biblioteca Spotify Full (1,18 milhão de faixas).

## 🚀 Modelos Implementados
* **Random Forest** (ajustado com pesos balanceados e limiar calibrado)
* **XGBoost** (Hist Gradient Boosting com scale_pos_weight — melhor ROC-AUC 0.8864)
* **TensorFlow MLP** (Rede Neural Profunda com Dropout e Regularização)

## 📦 Como Rodar Localmente

```bash
# Instalar dependências
pip install -r requirements.txt

# Iniciar servidor
python app.py
```

O site estará disponível em `http://localhost:7860`.
