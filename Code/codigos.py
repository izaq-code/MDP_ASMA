# -*- coding: utf-8 -*-
# ===========================================================
# PNS 2019 — Usa TODAS as numéricas (exceto Q074) e IGNORA Q075
# Anti-vazamento: remove 100% NaN, quase constantes, idênticas ao alvo e |r|>=0.98
# Pipeline: Imputer -> Z-Score -> SMOTE (apenas no treino) -> BayesSearchCV
# Saídas: metricas_modelos.csv, entropia_features.csv,
#         comparacao_modelos_metrics.png (com rótulos),
#         tsne_balanceado.png
# ===========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
)
from sklearn.manifold import TSNE

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from skopt import BayesSearchCV
from skopt.space import Integer, Categorical

# ----------------- PARÂMETROS -----------------
DATA_PATH = r"C:\Users\isaqu\OneDrive\Desktop\MDP_AVC\Data\pns_2019_recorte_final.csv"
ALVO = "Q074"          # 1 = tem asma (positiva), 0 = não tem (saudável)
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Excluir de TODAS as análises de features (além do alvo): Q075
EXCLUDE_FEATURES = {"Q075"}

# ----------------- LOAD -----------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Base não encontrada: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, sep=",", low_memory=False)
df = df.replace([np.inf, -np.inf], np.nan)

if ALVO not in df.columns:
    raise ValueError(f"Coluna alvo '{ALVO}' não está na base.")

# força numérico em todas as colunas possíveis
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# define y e remove alvo do DF de features
y = (df[ALVO] == 1).astype(int)
df_features = df.drop(columns=[ALVO])

# ----------------- CONSTRUÇÃO ROBUSTA DE X -----------------
def build_X(features_df: pd.DataFrame, exclude: set) -> pd.DataFrame:
    # 1) apenas numéricas e fora da lista de exclusão
    num_cols = [c for c in features_df.columns
                if pd.api.types.is_numeric_dtype(features_df[c]) and c not in exclude]
    X_ = features_df[num_cols].copy()

    # 2) remove 100% NaN
    X_ = X_.dropna(axis=1, how="all")

    # 3) remove quase constantes (>= 99,5% do mesmo valor não-nulo)
    def is_quasi_constant(s: pd.Series, thr=0.995) -> bool:
        s2 = s.dropna()
        if s2.empty:
            return True
        return (s2.value_counts(normalize=True).iloc[0] >= thr)

    keep = [c for c in X_.columns if not is_quasi_constant(X_[c], thr=0.995)]
    X_ = X_[keep]

    # Fallbacks progressivos se zerar:
    if X_.shape[1] == 0:
        print("[WARN] Todas as colunas caíram no filtro de constância. Relaxando para 99%…")
        num_cols_relax = [c for c in features_df.columns
                          if pd.api.types.is_numeric_dtype(features_df[c]) and c not in exclude]
        keep = [c for c in num_cols_relax if not is_quasi_constant(features_df[c], thr=0.99)]
        X_ = features_df[keep].copy()
        if X_.shape[1] == 0:
            print("[WARN] Ainda vazio. Mantendo as TOP 60 colunas com mais valores não-nulos.")
            nn = features_df[num_cols_relax].notna().sum().sort_values(ascending=False)
            top = [c for c in nn.index.tolist() if c not in exclude][:60]
            X_ = features_df[top].copy()

    return X_

X = build_X(df_features, EXCLUDE_FEATURES)

# 4) remover linhas sem alvo
mask = y.notna()
X, y = X[mask], y[mask]

# 5) Anti-vazamento: idênticas ao alvo e correlação alta com alvo
if X.shape[1] > 0:
    # imputação provisória para checar correlação
    _tmp_imp = SimpleImputer(strategy="median").fit_transform(X)
    _tmpX = pd.DataFrame(_tmp_imp, columns=X.columns, index=X.index)

    # 5.1) remove colunas idênticas ao alvo (ponto-a-ponto)
    identicas = [c for c in _tmpX.columns if _tmpX[c].equals(y)]
    if identicas:
        print(f"[INFO] Removendo colunas idênticas ao alvo: {identicas}")
        X = X.drop(columns=identicas)
        _tmpX = _tmpX.drop(columns=identicas)

    # 5.2) remove |r| >= 0.98 com alvo (se ainda houver cols)
    if _tmpX.shape[1] >= 5:
        corr_target = _tmpX.assign(target=y.values).corr(numeric_only=True)["target"].abs().drop("target")
        leakers = corr_target[corr_target >= 0.98].index.tolist()
        if leakers:
            print(f"[INFO] Removendo colunas com correlação >= 0.98 com o alvo: {leakers}")
            X = X.drop(columns=leakers)

if X.shape[1] == 0:
    raise ValueError("Após os filtros (NaN, constância, leakage), não restaram features. "
                     "Relaxe os limiares ou amplie o recorte da base.")

# prints diagnósticos úteis
print(f"\n[INFO] X shape final (sem Q075): {X.shape}; y classes: {y.value_counts().to_dict()}")
print("\n[INFO] Variância mínima das features (top 10):")
print(X.var(skipna=True).sort_values().head(min(10, X.shape[1])))

# ----------------- SPLIT -----------------
if y.nunique() < 2:
    raise ValueError("O alvo ficou com uma classe só. Ajuste o recorte.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

# ----------------- ENTROPIA (no treino imputado) -----------------
imp_entropy = SimpleImputer(strategy="median").fit(X_train)
Xtr_imp = pd.DataFrame(imp_entropy.transform(X_train), columns=X_train.columns)

def shannon_entropy(vec, bins=24):
    hist, _ = np.histogram(vec, bins=bins)
    tot = hist.sum()
    if tot == 0: return 0.0
    p = hist[hist > 0] / tot
    return float(-(p * np.log2(p)).sum())

entropy_df = pd.DataFrame({
    "feature": Xtr_imp.columns,
    "entropy_bits": [shannon_entropy(Xtr_imp[c].values, 24) for c in Xtr_imp.columns]
}).sort_values("entropy_bits", ascending=False)
# (Q075 já não está em X; portanto, também não aparecerá aqui)
entropy_df.to_csv("entropia_features.csv", index=False)

# ----------------- PIPE + BAYES -----------------
def make_pipe(clf):
    return ImbPipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE, sampling_strategy="auto")),
        ("clf", clf)
    ])

search_spaces = {
    "DecisionTree": {
        "clf__criterion": Categorical(["gini", "entropy", "log_loss"]),
        "clf__max_depth": Integer(2, 30),
        "clf__min_samples_split": Integer(2, 30),
        "clf__min_samples_leaf": Integer(1, 20)
    },
    "KNN": {
        "clf__n_neighbors": Integer(3, 35),
        "clf__weights": Categorical(["uniform", "distance"]),
        "clf__p": Categorical([1, 2])
    },
    "RandomForest": {
        "clf__n_estimators": Integer(150, 600),
        "clf__max_depth": Integer(3, 40),
        "clf__min_samples_split": Integer(2, 30),
        "clf__min_samples_leaf": Integer(1, 20),
        "clf__max_features": Categorical(["sqrt", "log2"])
    }
}

models = {
    "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(),
    "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
}

def fit_bayes(name, base_clf):
    opt = BayesSearchCV(
        estimator=make_pipe(base_clf),
        search_spaces=search_spaces[name],
        n_iter=25, cv=5, scoring="f1",
        n_jobs=-1, random_state=RANDOM_STATE, verbose=0
    )
    opt.fit(X_train, y_train)
    return opt

# ----------------- TREINO & AVALIAÇÃO -----------------
results = {}
for name, clf in models.items():
    print(f"\n=== {name} — BayesSearchCV ===")
    opt = fit_bayes(name, clf)

    y_pred = opt.predict(X_test)
    y_proba = opt.predict_proba(X_test)[:,1] if hasattr(opt, "predict_proba") else None
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    results[name] = {"accuracy":acc,"precision":prec,"recall":rec,"f1":f1,"roc_auc":auc}

    print("Melhores hiperparâmetros:", opt.best_params_)
    print("Melhor F1 (CV):", round(opt.best_score_, 4))
    print("Teste -> Acc:", round(acc,4), "Prec:", round(prec,4),
          "Rec:", round(rec,4), "F1:", round(f1,4), "AUC:", (round(auc,4) if not np.isnan(auc) else "NA"))
    print("\nRelatório de classificação (teste):\n",
          classification_report(y_test, y_pred, digits=4, zero_division=0))

# ----------------- MÉTRICAS + PNG com RÓTULOS -----------------
metrics_df = pd.DataFrame.from_dict(results, orient="index").reset_index().rename(columns={"index":"model"}).round(4)
metrics_df.to_csv("metricas_modelos.csv", index=False)

plt.figure(figsize=(10,5))
metric_names = ["accuracy","precision","recall","f1","roc_auc"]
x = np.arange(len(metrics_df)); width = 0.16
ax = plt.gca()
for i, m in enumerate(metric_names):
    vals = metrics_df[m].tolist()
    bars = ax.bar(x + (i-2)*width, vals, width, label=m)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.3f}", (b.get_x()+b.get_width()/2, b.get_height()),
                    xytext=(0,3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(metrics_df["model"])
ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
ax.set_title("Comparação — (todas numéricas exceto Q074, ignorando Q075) + SMOTE + BayesSearchCV")
ax.legend(ncols=3, fontsize=8)
plt.tight_layout(); plt.savefig("comparacao_modelos_metrics.png", dpi=300); plt.close()

# ----------------- t-SNE (treino balanceado) -----------------
imp = SimpleImputer(strategy="median").fit(X_train)
sc  = StandardScaler().fit(imp.transform(X_train))
Xtr_scaled = sc.transform(imp.transform(X_train))
Xbal, ybal = SMOTE(random_state=RANDOM_STATE, sampling_strategy="auto").fit_resample(Xtr_scaled, y_train)

max_pts = 1500
if len(ybal) > max_pts:
    idx = np.random.default_rng(123).choice(len(ybal), size=max_pts, replace=False)
    Xtsne_in, ytsne_in = Xbal[idx], ybal.iloc[idx].to_numpy()
else:
    Xtsne_in, ytsne_in = Xbal, ybal.to_numpy()

tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", n_iter=500, random_state=RANDOM_STATE)
Xtsne = tsne.fit_transform(Xtsne_in)

plt.figure(figsize=(6.5,6.5))
m0, m1 = (ytsne_in==0), (ytsne_in==1)
plt.scatter(Xtsne[m0,0], Xtsne[m0,1], s=6, alpha=0.65, label=f"Não asmático (n={m0.sum()})")
plt.scatter(Xtsne[m1,0], Xtsne[m1,1], s=6, alpha=0.65, label=f"Asmático (n={m1.sum()})")
plt.title("t-SNE do Treino Balanceado (Imputer + Z-Score + SMOTE) — sem Q075")
plt.legend(markerscale=2); plt.tight_layout(); plt.savefig("tsne_balanceado.png", dpi=300); plt.close()

print("\nArquivos gerados:")
print("- comparacao_modelos_metrics.png")
print("- tsne_balanceado.png")
print("- metricas_modelos.csv")
print("- entropia_features.csv")
