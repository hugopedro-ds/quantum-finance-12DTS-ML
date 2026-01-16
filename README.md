# Quantum Finance — credit-score-classification-machine-learning (FIAP)

---

# 🏷️ Badges

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Notebook](https://img.shields.io/badge/notebook-Jupyter%2FColab-red)
![License](https://img.shields.io/badge/license-academic-lightgrey)

---

# 📋 Visão Geral
Projeto integrado de Machine Learning - Quantum Finance (FIAP) usando o dataset **Credit Score Classification (Kaggle)** para prever a classe de risco de crédito (**Good / Standard / Poor**) e traduzir resultados em **decisão de negócio**.

---

## 📁 Estrutura do projeto

```
img/                     # Gráficos, tabelas e outputs do modelo
├── 03_dataset_preview_head_1.png
├── 08_model_comparison_table.png
├── 09_feature_importance_top15_bar.png
└── ...

notebooks/               # Notebook principal do projeto
└── Trabalho_de_MLearning.ipynb

README.md                # Documentação do projeto
outputs/                 # Resultados gerados (previsões, métricas, relatórios)
models/                  # Modelos treinados (se aplicável)
```

---

# 🧠 Técnicas e Abordagem

- **Modelagem supervisionada (classificação multiclasse)** para prever `Credit_Score` (**Good / Standard / Poor**) a partir de variáveis financeiras e comportamentais.
- **Prevenção de leakage (cenário realista)** com **split por cliente** via `GroupShuffleSplit` (o mesmo cliente não aparece em treino e teste).
- **Pré-processamento robusto em pipeline** com `ColumnTransformer`:
  - **Numéricas:** imputação (mediana) + `StandardScaler`
  - **Categóricas:** imputação (moda) + `OneHotEncoder(handle_unknown="ignore")`
- **Comparação de modelos** com execução padronizada via pipelines:
  - Logistic Regression, Decision Tree, Random Forest, HistGradientBoosting (**seleção pelo melhor desempenho**)
- **Métrica principal:** **F1-macro**, adequada para multiclasse e possível desbalanceamento.
- **Explicabilidade** com **Permutation Importance**, destacando os **Top 15 drivers** de risco.
- **Segunda abordagem explicável (Sistema Especialista)** com regras/pontuação + *reasons*, comparada ao ML.
- **Saída orientada à decisão de negócio**, combinando predição + probabilidades + razões + ação recomendada.

---

# 🚀 Como Usar

### 1) Instalação (requirements.txt)
- **Pré-requisitos**
  - Python **3.8+**
  - Git (opcional)
- **Instalar dependências**
  - `pip install -r requirements.txt`
 
### 2) Reprodutibilidade
- As dependências do projeto estão descritas em `requirements.txt` com versões fixas.
- Para garantir reprodutibilidade, recomenda-se executar o projeto em ambiente limpo.
- O notebook principal concentra todo o fluxo de dados, treino, avaliação e outputs.

### 3) Rodar o projeto
- **Opção A — Google Colab (recomendado)**
  - Abrir o notebook: **Open in Colab**
  - Executar as células em ordem (Runtime → Run all)

- **Opção B — Local**
  - Abrir `notebooks/Trabalho_de_MLearning.ipynb` no Jupyter ou VS Code
  - Garantir dependências instaladas: `pip install -r requirements.txt`

### 4) Outputs gerados
- **Imagens (para leitura rápida no GitHub)**
  - Pasta: `img/`
  - Exemplos:
    - `08_model_comparison_table.png` — comparação de modelos
    - `09_feature_importance_top15_bar.png` — drivers principais (Permutation Importance)

---

## 📊 Dataset
- Fonte: Credit Score Classification (Kaggle)
- O notebook faz download automaticamente via `kagglehub`
- Obs.: para executar (Colab ou local), pode ser necessário configurar credenciais do Kaggle (kaggle.json) no ambiente

![Dataset preview](assets/03_dataset_preview_head_1.png)

---

## 🔎 O que foi feito no notebook

### 1) Importação e leitura da base (Kaggle)
- Download via `kagglehub`
- Leitura do `train.csv` com `pandas`

### 2) EDA — Análise Exploratória
- Head / info / tipos
- Distribuição do target (`Credit_Score`)
- Análise de missing (top colunas com faltantes)

### 3) Limpeza e padronização de variáveis
- Conversão robusta de colunas “numéricas sujas” (ex.: símbolos, textos e valores inválidos)
- Transformação de `Credit_History_Age` para **meses** (`Credit_History_Age_Months`)
- Remoção de colunas de identificação e baixo valor de negócio

### 4) Split correto por cliente (sem vazamento)
- `GroupShuffleSplit(test_size=0.2)`
- Remoção de `Customer_ID` das features

### 5) Pipeline de pré-processamento
- Numéricas: imputação por mediana + **StandardScaler**
- Categóricas: imputação por moda + **OneHotEncoder**
- Montagem via `ColumnTransformer`

### 6) Treino e comparação dos modelos
- Execução padronizada por pipeline
- Coleta de métricas por modelo
- Seleção do melhor pelo **F1-macro**

### 7) Interpretação do modelo (feature importance)
- **Permutation Importance** (efeito real no desempenho)
- Gráfico Top 15 para comunicação em slide

### 8) Sistema Especialista (Regras) + avaliação
- Regras interpretáveis (pontuação por risco)
- Matriz de confusão + métricas (Accuracy, F1-macro)
- Comparação final: **ML vs Regras**

### 9) Saída final orientada a negócio
- Classe final (ML por default)
- Probabilidades do ML (quando disponível)
- Motivos das regras (reasons)
- Ação recomendada (aprovar / ajustar / revisar)

---

## 🏆 Resultados (Resumo Executivo)
**Melhor modelo (supervisionado):** `HistGradientBoosting (HistGB)`  
- **F1-macro:** ~ **0,685**
- **Accuracy:** ~ **0,705**

📌 Métricas (melhor modelo)
| Modelo | Accuracy | F1-macro |
|---|---:|---:|
| HistGradientBoosting (HistGB) | 0.705 | 0.685 |

**Comparação de modelos**

![Model Comparison](assets/08_model_comparison_table.png)

![HistGB report](assets/07_model_histgb_report.png)

![LogReg report](assets/04_model_logreg_report.png)

![DecisionTree report](assets/05_model_decisiontree_report.png)

![RandomForest report](assets/06_model_randomforest_report.png)

**Interpretação (drivers principais de risco):**
- `Credit_Mix`
- `Outstanding_Debt`
- `Interest_Rate`
- `Delay_from_due_date`
- `Changed_Credit_Limit`
- `Payment_of_Min_Amount`
- `Credit_History_Age_Months`

**Drivers principais de risco (Permutation Importance)**
![Feature Importance](assets/09_feature_importance_top15_bar.png)

![Feature importance table](assets/10_feature_importance_table.png)

---

## 💡 Insights em Visão de Negócio
### 📉 Padrões que tendem a elevar risco
- **Pior mix de crédito** (indicador de perfil menos saudável)
- **Maior endividamento** e **juros altos**
- **Atrasos** e comportamento de pagamento mínimo
- **Instabilidade** (mudanças de limite, histórico curto, exposição alta)

### 🎯 Ação recomendada por classe
- **Good:** Aprovar automaticamente (condições melhores / limite padrão)
- **Standard:** Aprovar com **limite menor** / **taxa ajustada** / validações adicionais
- **Poor:** Encaminhar para **análise manual** ou reprovar conforme política de risco

---

## ✅ Entregáveis

✅ **Entregável 1 — Contexto e Objetivo**
- Problema: predição de risco de crédito em 3 classes
- Valor: decisões mais rápidas, consistentes e auditáveis

✅ **Entregável 2 — Protótipo Operacional**
- Notebook executável com pipeline completo
- Outputs salvos para leitura no GitHub

✅ **Entregável 3 — Avaliação e Explicabilidade**
- Métricas padronizadas
- Matriz de confusão e report

![Rules confusion matrix](assets/17_rules_confusion_matrix.png)
  
- Permutation Importance + gráfico Top 15

✅ **Entregável 4 — Segunda Abordagem**
- Sistema Especialista (regras) com razões
- Comparação com ML

![ML vs Rules](assets/18_ml_vs_rules_metrics_comparison.png)

---

## 📑 Critérios de Avaliação (Rubrica)
- [x] Definição clara do problema de negócio
- [x] Análise exploratória dos dados (EDA)
- [x] Prevenção de data leakage
- [x] Pipeline de pré-processamento
- [x] Treino e comparação de modelos
- [x] Métricas adequadas ao problema
- [x] Explicabilidade do modelo
- [x] Tradução dos resultados em decisão de negócio

---

## 🔧 Funcionalidades
- Treina e avalia modelos de classificação de risco
- Interpreta variáveis mais relevantes (explicabilidade)
- Executa sistema explicável por regras
- Gera recomendações de decisão (ação por classe)
- Mostra exemplos práticos com “reasons” para auditoria

---

## 📈 Exemplo de Saída
📌 **Cliente (amostra)**  
- `ML Pred:` Standard  
- `Probabilidades:` Good 0.12 | Standard 0.71 | Poor 0.17  
- `Regras:` Standard (5 pontos)  
- `Reasons:` Credit_Mix=Bad (+3) | Delay>15 (+2)  

✅ **Ação:** aprovar com limite menor + validações adicionais

![Case study](assets/19_case_study_10_examples_with_reasons.png)

---

## 🔐 Segurança e Privacidade
- Evitamos uso de identificadores diretos (`SSN`, `Name`, etc.)
- Split por cliente para simular cenário real
- Abordagem explicável (regras) para auditoria e governança

---

## 📦 Dependências

- Python: 3.8+
- NumPy: cálculos numéricos
- Pandas: manipulação de dados
- Scikit-learn: pipelines, modelos, métricas e avaliação
- Matplotlib: visualizações gráficas
- Seaborn: visualizações exploratórias (EDA)
- kagglehub: download automático do dataset

---

## 🛣️ Próximos Passos (Evolução realista)
- **Tuning** (GridSearch/RandomSearch) no HistGB
- **Calibração de probabilidades** (Platt / isotônica) para decisões de limite/taxa
- **Explainability avançada:** SHAP (em subset) + relatórios
- **Tratamento robusto de outliers e variáveis sujas**
- **Deploy de prova de conceito:** API (FastAPI) com endpoint `/score`
- **Monitoramento:** drift de dados e performance + revisão periódica
- **Política de crédito:** mapear score → limite/taxa com regras de negócio

---

## 📚 Referências
- Dataset: Credit Score Classification (Kaggle)
- Scikit-learn: Pipelines, ColumnTransformer, GroupShuffleSplit, Permutation Importance
- Boas práticas: prevenção de leakage e avaliação com F1-macro

---
## 📄 Licença
Este projeto é uma prova de conceito desenvolvida para fins acadêmicos.

---

## 👥 Autores
Projeto desenvolvido para o Case Study de Machine Learning (FIAP).
