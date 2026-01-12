# Quantum Finance — Machine Learning (FIAP)
Projeto integrado de Machine Learning - Quantum Finance (FIAP) usando o dataset **Credit Score Classification (Kaggle)** para prever a classe de risco de crédito (**Good / Standard / Poor**) e traduzir resultados em **decisão de negócio**.

**O que foi entregue:**
- **EDA** (exploração + qualidade dos dados + missing)
- **Split correto por cliente** com `GroupShuffleSplit` usando `Customer_ID` (evita vazamento de dados)
- **Pré-processamento** (limpeza, imputação, One-Hot + padronização)
- **Modelos supervisionados** comparados (LogReg, DecisionTree, RandomForest, HistGradientBoosting)
- **Avaliação** (Accuracy, F1-macro, matriz de confusão e relatório)
- **Interpretação** com *Permutation Importance* (Top drivers do score)
- **2ª abordagem (explicável): Sistema Especialista (Regras)** + comparação vs modelo ML
- **Saída final orientada a negócio** (classe + probabilidade + razões + ação recomendada)

## 🚀 Abrir no Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dbZLeA2FUk-m8vub7CACAM08gGH4hQ1G?usp=sharing)

- **Notebook principal:** `notebooks/Trabalho_de_MLearning.ipynb`

## Principais insights (visão de negócio)
O risco tende a aumentar em perfis com:
- **Pior mix de crédito (Credit_Mix)**,
- **Maior endividamento (Outstanding_Debt)**,
- **Juros mais altos (Interest_Rate)**,
- **Atrasos (Delay_from_due_date)**,
- E sinais de instabilidade (ex.: mudanças relevantes no limite de crédito).

**Ação recomendada:**
- `Good`: aprovar automático (condições melhores)
- `Standard`: aprovar com limite/taxa ajustados + validações adicionais
- `Poor`: análise manual ou reprovação conforme política de risco

## Reprodutibilidade
O notebook foi salvo com **outputs (tabelas e gráficos)** para facilitar avaliação e leitura no GitHub.

