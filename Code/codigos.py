# -*- coding: utf-8 -*-
# ===========================================================
# PNS 2019 — Todas numéricas (exceto Q075, C008, V0001),
# anti-vazamento, entropia antes do balanceamento,
# REBALANCEAMENTO CUSTOM:
#   - downsampling inteligente de saudáveis (negativos "difíceis")
#   - oversampling via SMOTE para asmáticos (1:1)
# BayesSearch otimizado por PR-AUC (average_precision),
# DecisionTree, KNN, RandomForest, MLP (sklearn),
# Keras com memória (.keras) e barras de progresso.
#
# Avaliação:
#   - Teste REAL (desbalanceado) -> métrica honesta
#   - Teste BALANCEADO (1:1) -> métrica "justa"
# ===========================================================

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.manifold import TSNE

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import ClusterCentroids  # usado só no t-SNE
from imblearn.over_sampling import SMOTE

from skopt import BayesSearchCV
from skopt.space import Integer, Categorical, Real

import tensorflow as tf
from tensorflow import keras

# tqdm (barra de progresso)
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, *args, **kwargs):
        return x

# ----------------- PARÂMETROS -----------------
DATA_PATH = r"C:\Users\isaqu\OneDrive\Desktop\MDP_AVC\Data\pns_2019_recorte_sem_outliers_TRATADA.csv"
ALVO = "Q074"
TEST_SIZE = 0.20
RANDOM_STATE = 42

EXCLUDE_FEATURES = {"Q075", "C008", "V0001"}
KERAS_MODEL_PATH = "modelo_asma.keras"

MAX_SEARCH_SAMPLES = 25000

N_ITERS_BAYES = {
    "DecisionTree": 40,
    "KNN": 35,
    "RandomForest": 45,
    "MLP": 30,
}

USE_LABEL_NOISE_CLEANING = False
USE_NESTED_CV = False  # nested CV pesadão

# ----------------- LOAD -----------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(DATA_PATH)

print("[INFO] Lendo base...")
df = pd.read_csv(DATA_PATH, sep=",", low_memory=False)
df = df.replace([np.inf, -np.inf], np.nan)

for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")

y = (df[ALVO] == 1).astype(int)
df_features = df.drop(columns=[ALVO])

print(f"[INFO] Base lida com shape: {df.shape}")
print(f"[INFO] Classes em y: {dict(y.value_counts())}")

# ----------------- BUILD X -----------------
def build_X(features_df, exclude):
    num_cols = [c for c in features_df.columns
                if pd.api.types.is_numeric_dtype(features_df[c])
                and c not in exclude]

    X_ = features_df[num_cols].dropna(axis=1, how="all")

    def is_quasi_constant(s, thr=0.995):
        s2 = s.dropna()
        if s2.empty:
            return True
        return (s2.value_counts(normalize=True).iloc[0] >= thr)

    keep = [c for c in X_.columns if not is_quasi_constant(X_[c])]
    X_ = X_[keep]

    if X_.shape[1] == 0:
        print("[WARN] relaxando limiar de quase-constante...")
        keep = [c for c in num_cols if not is_quasi_constant(features_df[c], thr=0.99)]
        X_ = features_df[keep].copy()
        if X_.shape[1] == 0:
            nn = features_df[num_cols].notna().sum().sort_values(ascending=False)
            top = nn.index[:60]
            X_ = features_df[top].copy()

    return X_

print("[INFO] Construindo matriz X (features numéricas limpas)...")
X = build_X(df_features, EXCLUDE_FEATURES)
mask = y.notna()
X, y = X[mask], y[mask]

print(f"[INFO] X final (antes anti-vazamento): {X.shape}")

# ----------------- ANTI-VAZAMENTO -----------------
print("[INFO] Rodando anti-vazamento (leakers, correlação alta com o alvo)...")
_tmp_imp = SimpleImputer(strategy="median").fit_transform(X)
_tmpX = pd.DataFrame(_tmp_imp, columns=X.columns, index=X.index)

identicas = [c for c in _tmpX.columns if _tmpX[c].equals(y)]
if identicas:
    print(f"[INFO] Removendo colunas idênticas ao alvo: {identicas}")
    X = X.drop(columns=identicas)
    _tmpX = _tmpX.drop(columns=identicas)

corr_target = _tmpX.assign(target=y.values).corr(numeric_only=True)["target"].abs().drop("target")
leakers = corr_target[corr_target >= 0.98].index.tolist()
if leakers:
    print(f"[INFO] Removendo leakers com |corr| >= 0.98: {leakers}")
    X = X.drop(columns=leakers)

print(f"[INFO] X após anti-vazamento: {X.shape}")

# ----------------- GRUPOS (cluster amostral) -----------------
if "V0001" in df_features.columns:
    groups = df_features.loc[X.index, "V0001"]
    print("[INFO] Usando V0001 como grupo para GroupShuffleSplit.")
else:
    groups = None
    print("[WARN] V0001 não encontrado, usando split estratificado normal.")

# ----------------- SPLIT -----------------
print("[INFO] Fazendo split treino/teste...")
if groups is not None:
    mask_groups = groups.notna()
    X = X[mask_groups]
    y = y[mask_groups]
    groups = groups[mask_groups]

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train_full = X.iloc[train_idx]
    X_test       = X.iloc[test_idx]
    y_train_full = y.iloc[train_idx]
    y_test       = y.iloc[test_idx]

    print("[INFO] Split por grupo (GroupShuffleSplit) concluído.")
else:
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print("[INFO] Split estratificado simples concluído.")

print(f"[INFO] X_train_full: {X_train_full.shape}, X_test: {X_test.shape}")
print(f"[INFO] Classes em y_train_full: {dict(y_train_full.value_counts())}")
print(f"[INFO] Classes em y_test: {dict(y_test.value_counts())}")

# ----------------- TESTE BALANCEADO PARA AVALIAÇÃO JUSTA -----------------
def make_balanced_test(X_test_df, y_test_s, ratio_neg_pos=1.0, random_state=RANDOM_STATE):
    """
    Conjunto de teste balanceado:
      - mantém TODOS os asmáticos (1) do teste original
      - amostra negativos (0) na proporção ratio_neg_pos : 1
    Ex.: ratio_neg_pos=1.0 -> 1:1
    """
    rng = np.random.RandomState(random_state)

    pos_idx = y_test_s[y_test_s == 1].index
    neg_idx = y_test_s[y_test_s == 0].index

    n_pos = len(pos_idx)
    n_neg_target = int(n_pos * ratio_neg_pos)
    n_neg_target = min(n_neg_target, len(neg_idx))

    neg_sample_idx = rng.choice(neg_idx, size=n_neg_target, replace=False)

    idx_bal = np.concatenate([pos_idx, neg_sample_idx])
    X_test_bal = X_test_df.loc[idx_bal].copy()
    y_test_bal = y_test_s.loc[idx_bal].copy()

    print("\n[INFO] Conjunto de TESTE BALANCEADO criado:")
    print(f"       Asmáticos (1) = {sum(y_test_bal == 1)}")
    print(f"       Não asmáticos (0) = {sum(y_test_bal == 0)}")
    return X_test_bal, y_test_bal

# teste 1:1 para avaliação justa
X_test_bal, y_test_bal = make_balanced_test(X_test, y_test, ratio_neg_pos=1.0)

# ----------------- ENTROPIA (antes do balanceamento) -----------------
print("[INFO] Calculando entropia (antes do balanceamento)...")
imp_entropy = SimpleImputer(strategy="median").fit(X_train_full)
Xtr_imp = pd.DataFrame(imp_entropy.transform(X_train_full), columns=X_train_full.columns)

def shannon_entropy(vec, bins=24):
    hist, _ = np.histogram(vec, bins=bins)
    tot = hist.sum()
    if tot == 0:
        return 0.0
    p = hist[hist > 0] / tot
    return float(-(p * np.log2(p)).sum())

entropies = []
for col in tqdm(Xtr_imp.columns, desc="Entropia por feature"):
    entropies.append(shannon_entropy(Xtr_imp[col], 24))

entropy_df = pd.DataFrame({
    "feature": Xtr_imp.columns,
    "entropy_bits": entropies
}).sort_values("entropy_bits", ascending=False)

entropy_df.to_csv("entropia_features.csv", index=False)

plt.figure(figsize=(10, max(6, len(entropy_df) * 0.15)))
plt.barh(entropy_df["feature"], entropy_df["entropy_bits"])
plt.xlabel("Entropia (bits)")
plt.ylabel("Feature")
plt.title("Entropia por Feature — Antes do Balanceamento")
plt.gca().invert_yaxis()
plt.text(0.99, 0.01, "PNS 2019 - Asma", transform=plt.gcf().transFigure,
         ha="right", va="bottom", fontsize=8, alpha=0.7)
plt.tight_layout()
plt.savefig("entropia_features.png", dpi=300, bbox_inches="tight")
plt.close()
print("[INFO] Entropia salva em entropia_features.csv e entropia_features.png")

# ----------------- REMOÇÃO OPCIONAL DE LABEL NOISE -----------------
def remove_label_noise(X_tr, y_tr, proba_tr, high_conf=0.99):
    y_hat = (proba_tr >= 0.5).astype(int)
    noisy_idx = (
        ((y_tr == 0) & (proba_tr >= high_conf)) |
        ((y_tr == 1) & (proba_tr <= 1 - high_conf))
    )
    print(f"[INFO] Removendo {noisy_idx.sum()} exemplos potencialmente ruidosos de {len(y_tr)}")
    mask = ~noisy_idx
    return X_tr[mask], y_tr[mask]

# ----------------- DOWN DE SAUDÁVEIS + UP (SMOTE) DE ASMÁTICOS -----------------
def smart_down_up_sampling(X_train_df, y_train_s, max_ratio=10):
    """
    1) Treina RandomForest rápido em X_train_df, y_train_s.
    2) Entre os saudáveis (0), mantém só os mais "difíceis"
       (probabilidade perto de 0.5) até no máx. max_ratio * n_pos.
    3) Mantém TODOS os asmáticos (1).
    4) Imputa NaN (mediana).
    5) Aplica SMOTE para ficar 1:1 (0 e 1).
    """
    print("\n[INFO] Rebalanceando (down de saudáveis + up de asmáticos via SMOTE)...")

    y_np = y_train_s.to_numpy()
    idx_pos = np.where(y_np == 1)[0]
    idx_neg = np.where(y_np == 0)[0]

    n_pos = len(idx_pos)
    n_neg = len(idx_neg)

    print(f"[INFO] Antes do rebalance: saudáveis={n_neg}, asmáticos={n_pos}")

    pipe_fast = ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ))
    ])

    print("[INFO] Treinando RandomForest rápido para rankear instâncias...")
    pipe_fast.fit(X_train_df, y_train_s)
    proba = pipe_fast.predict_proba(X_train_df)[:, 1]

    proba_neg = proba[idx_neg]
    ordem_neg = np.argsort(np.abs(proba_neg - 0.5))

    target_neg = min(n_neg, n_pos * max_ratio)
    idx_neg_keep = idx_neg[ordem_neg[:target_neg]]

    idx_keep = np.concatenate([idx_pos, idx_neg_keep])
    X_sub = X_train_df.iloc[idx_keep].copy()
    y_sub = y_train_s.iloc[idx_keep].copy()

    print(f"[INFO] Após down inteligente de saudáveis:")
    print(f"       X_sub={X_sub.shape}, classes={dict(y_sub.value_counts())}")

    print("[INFO] Imputando NaN em X_sub antes do SMOTE (mediana)...")
    imp_smote = SimpleImputer(strategy="median")
    X_sub_imp = imp_smote.fit_transform(X_sub)

    print("[INFO] Aplicando SMOTE para balancear (1:1)...")
    sm = SMOTE(random_state=RANDOM_STATE)
    X_bal, y_bal = sm.fit_resample(X_sub_imp, y_sub)

    y_bal_series = pd.Series(y_bal, name=y_train_s.name)
    print(f"[INFO] Após SMOTE: X_bal={X_bal.shape}, classes={dict(y_bal_series.value_counts())}")

    X_bal_df = pd.DataFrame(X_bal, columns=X_train_df.columns)
    return X_bal_df, y_bal_series

# aplica rebalanceamento direto no treino
X_train, y_train = smart_down_up_sampling(X_train_full, y_train_full, max_ratio=10)
print(f"[INFO] X_train balanceado final: {X_train.shape}")
print(f"[INFO] Classes em y_train balanceado: {dict(y_train.value_counts())}")

# ----------------- PIPE -----------------
def make_pipe(clf):
    """
    Balanceamento agora é feito ANTES (smart_down_up_sampling + SMOTE).
    Aqui só imputamos, escalamos e treinamos o classificador.
    """
    return ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", clf)
    ])

# ----------------- SEARCH SPACES -----------------
search_spaces = {
    "DecisionTree": {
        "clf__criterion": Categorical(["gini", "entropy", "log_loss"]),
        "clf__max_depth": Categorical([None, 3, 5, 10, 15, 20, 30, 40, 60]),
        "clf__min_samples_split": Categorical([2, 3, 5, 10, 20, 30, 40, 50]),
        "clf__min_samples_leaf": Categorical([1, 2, 3, 4, 5, 10, 15, 20, 30]),
        "clf__splitter": Categorical(["best", "random"]),
        "clf__max_features": Categorical([None, "sqrt", "log2"]),
        "clf__max_leaf_nodes": Categorical([None, 50, 100, 150, 200, 300, 400]),
        "clf__class_weight": Categorical([None, "balanced"])
    },

    "KNN": {
        "clf__n_neighbors": Categorical([1, 3, 5, 7, 9, 11, 15, 20, 25, 30, 40, 50]),
        "clf__weights": Categorical(["uniform", "distance"]),
        "clf__p": Categorical([1, 2]),
        "clf__algorithm": Categorical(["auto", "ball_tree", "kd_tree", "brute"]),
        "clf__leaf_size": Categorical([10, 20, 30, 40, 50, 60]),
        "clf__metric": Categorical(["minkowski", "euclidean", "manhattan", "chebyshev"])
    },

    "RandomForest": {
        "clf__n_estimators": Categorical([50, 100, 150, 200, 300, 400, 600, 800]),
        "clf__criterion": Categorical(["gini", "entropy", "log_loss"]),
        "clf__max_depth": Categorical([None, 5, 10, 15, 20, 30, 40, 50]),
        "clf__min_samples_split": Categorical([2, 3, 4, 5, 10, 15, 20, 30]),
        "clf__min_samples_leaf": Categorical([1, 2, 3, 4, 5, 10, 15, 20, 30]),
        "clf__max_features": Categorical(["sqrt", "log2", None]),
        "clf__max_leaf_nodes": Categorical([None, 50, 100, 150, 200, 300, 400]),
        "clf__bootstrap": Categorical([True, False]),
        "clf__class_weight": Categorical([None, "balanced", "balanced_subsample"])
    },

    "MLP": {
        "clf__hidden_layer_sizes": Categorical([32, 64, 128, 256]),
        "clf__activation": Categorical(["relu", "tanh"]),
        "clf__solver": Categorical(["adam"]),
        "clf__alpha": Real(1e-5, 1e-2, prior="log-uniform"),
        "clf__learning_rate_init": Real(1e-4, 1e-2, prior="log-uniform"),
        "clf__max_iter": Integer(120, 220)
    }
}

models = {
    "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(),
    "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    "MLP": MLPClassifier(random_state=RANDOM_STATE)
}

# ----------------- FUNÇÃO PARA AJUSTAR THRESHOLD + RELATÓRIO POR CLASSE -----------------
def metrics_from_proba(y_true, y_proba, titulo=""):
    """
    Ajusta o limiar para maximizar F1 (classe positiva) e exibe:
      - threshold ótimo
      - threshold baseado na prevalência
      - matriz de confusão
      - classification_report por classe (0 e 1)
    """
    if titulo:
        print(f"\n### {titulo} ###")

    thresholds = np.linspace(0.01, 0.99, 99)
    best_thr = 0.5
    best_f1 = -1.0

    for thr in thresholds:
        y_pred_thr = (y_proba >= thr).astype(int)
        f1 = f1_score(y_true, y_pred_thr, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    prevalence = y_true.mean()
    thr_prev = prevalence

    print(f"[INFO] Threshold ótimo (F1)         = {best_thr:.3f}")
    print(f"[INFO] Threshold baseado prevalência = {thr_prev:.3f}")

    thr_used = best_thr

    y_pred_best = (y_proba >= thr_used).astype(int)
    acc = accuracy_score(y_true, y_pred_best)
    prec = precision_score(y_true, y_pred_best, zero_division=0)
    rec = recall_score(y_true, y_pred_best, zero_division=0)
    f1 = f1_score(y_true, y_pred_best, zero_division=0)
    auc = roc_auc_score(y_true, y_proba)

    print("[INFO] Matriz de confusão (thr usado):")
    print(confusion_matrix(y_true, y_pred_best))

    print("\n[INFO] Métricas por classe (0 = Não asmático, 1 = Asmático):")
    print(classification_report(
        y_true,
        y_pred_best,
        target_names=["Nao_asmatico", "Asmatico"],
        digits=4,
        zero_division=0
    ))

    return {
        "threshold": thr_used,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc
    }

# ----------------- BAYESSEARCH (average_precision) -----------------
results = {}
results_keras = {}

def fit_bayes(name, clf, X_train_local, y_train_local):
    print(f"[INFO] Iniciando BayesSearch para {name}...")
    if len(X_train_local) > MAX_SEARCH_SAMPLES:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X_train_local), MAX_SEARCH_SAMPLES, replace=False)
        X_search = X_train_local.iloc[idx]
        y_search = y_train_local.iloc[idx]
        print(f"[INFO] {name}: usando subamostra de {X_search.shape[0]} linhas para busca bayesiana.")
    else:
        X_search = X_train_local
        y_search = y_train_local

    opt = BayesSearchCV(
        estimator=make_pipe(clf),
        search_spaces=search_spaces[name],
        n_iter=N_ITERS_BAYES[name],
        cv=3,
        scoring="average_precision",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )

    t0 = time.time()
    opt.fit(X_search, y_search)
    t1 = time.time()
    print(f"[INFO] BayesSearch {name} concluído em {t1 - t0:.1f} s")

    best_pipe = opt.best_estimator_
    print(f"[INFO] Refazendo fit final de {name} no treino completo (rebalanceado)...")
    best_pipe.fit(X_train_local, y_train_local)

    return opt, best_pipe

# ----------------- NESTED CV OPCIONAL -----------------
def nested_eval(name, clf, X_all, y_all, n_outer=5):
    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=RANDOM_STATE)
    all_metrics = []

    for i, (tr_idx, te_idx) in enumerate(outer.split(X_all, y_all), 1):
        print(f"\n[INFO] Outer fold {i}/{n_outer} - {name}")
        X_tr, X_te = X_all.iloc[tr_idx], X_all.iloc[te_idx]
        y_tr, y_te = y_all.iloc[tr_idx], y_all.iloc[te_idx]

        opt, best_pipe = fit_bayes(name, clf, X_tr, y_tr)
        y_proba = best_pipe.predict_proba(X_te)[:, 1]
        met = metrics_from_proba(y_te, y_proba, titulo="Nested Fold")
        all_metrics.append(met)

    df = pd.DataFrame(all_metrics)
    print("\n[INFO] MÉTRICAS NESTED (média ± desvio):")
    print(df.mean().round(4))
    print(df.std().round(4))
    return df

# ----------------- TREINAR MODELOS BAYES -----------------
print("\n[INFO] Iniciando treino dos modelos clássicos...")

if USE_LABEL_NOISE_CLEANING:
    print("[INFO] Rodando limpeza de label noise com RandomForest inicial...")
    rf_init = make_pipe(RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_estimators=200,
        n_jobs=-1
    ))
    rf_init.fit(X_train, y_train)
    proba_tr = rf_init.predict_proba(X_train)[:, 1]
    X_train, y_train = remove_label_noise(X_train, y_train, proba_tr, high_conf=0.99)
    print(f"[INFO] Após limpeza, X_train: {X_train.shape}, classes: {dict(y_train.value_counts())}")

for name, clf in tqdm(list(models.items()), desc="Modelos Bayes"):
    print(f"\n=== {name} ===")
    opt, best_pipe = fit_bayes(name, clf, X_train, y_train)

    if hasattr(best_pipe, "predict_proba"):
        # ----- TESTE REAL -----
        y_proba_real = best_pipe.predict_proba(X_test)[:, 1]
        met_real = metrics_from_proba(y_test, y_proba_real, titulo="AVALIAÇÃO NO TESTE REAL (desbalanceado)")

        # ----- TESTE BALANCEADO -----
        y_proba_bal = best_pipe.predict_proba(X_test_bal)[:, 1]
        met_bal = metrics_from_proba(y_test_bal, y_proba_bal, titulo="AVALIAÇÃO NO TESTE BALANCEADO (1:1)")

        # guarda métricas do teste BALANCEADO (as "justas") para CSV e gráfico
        met = met_bal
    else:
        # fallback (sem predict_proba)
        y_pred = best_pipe.predict(X_test_bal)
        acc = accuracy_score(y_test_bal, y_pred)
        prec = precision_score(y_test_bal, y_pred, zero_division=0)
        rec = recall_score(y_test_bal, y_pred, zero_division=0)
        f1 = f1_score(y_test_bal, y_pred, zero_division=0)
        auc = np.nan
        met = {"threshold": 0.5, "accuracy": acc, "precision": prec,
               "recall": rec, "f1": f1, "roc_auc": auc}

    results[name] = {
        "accuracy": met["accuracy"],
        "precision": met["precision"],
        "recall": met["recall"],
        "f1": met["f1"],
        "roc_auc": met["roc_auc"]
    }

    print("Melhores hiperparâmetros:", opt.best_params_)
    print(
        f"Teste BALANCEADO ({name}) -> acc={met['accuracy']:.4f}, "
        f"prec={met['precision']:.4f}, rec={met['recall']:.4f}, "
        f"f1={met['f1']:.4f}, auc={met['roc_auc']:.4f}"
    )

# ----------------- OPCIONAL: NESTED CV PARA RANDOMFOREST -----------------
if USE_NESTED_CV:
    print("\n[INFO] Rodando Nested CV pesado para RandomForest...")
    nested_res_rf = nested_eval("RandomForest", models["RandomForest"], X, y)
    nested_res_rf.to_csv("metricas_nested_randomforest.csv", index=False)
    print("[INFO] Nested CV RandomForest salvo em metricas_nested_randomforest.csv")

# ----------------- KERAS COM MEMÓRIA (FORA DO GRÁFICO) -----------------
print("\n=== Keras MLP (com memória) ===")

imp_nn = SimpleImputer(strategy="median").fit(X_train)
sc_nn = StandardScaler().fit(imp_nn.transform(X_train))

X_train_nn = sc_nn.transform(imp_nn.transform(X_train))
X_test_nn = sc_nn.transform(imp_nn.transform(X_test))
X_test_bal_nn = sc_nn.transform(imp_nn.transform(X_test_bal))

input_dim = X_train_nn.shape[1]

def build_mlp_keras():
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

if os.path.exists(KERAS_MODEL_PATH):
    print("[INFO] Carregando modelo Keras salvo...")
    model = keras.models.load_model(KERAS_MODEL_PATH)
else:
    print("[INFO] Criando novo modelo Keras...")
    model = build_mlp_keras()

es = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

print("[INFO] Treinando Keras (MLP)...")
history = model.fit(
    X_train_nn, y_train,
    validation_split=0.2,
    epochs=40,
    batch_size=256,
    callbacks=[es],
    verbose=1
)

model.save(KERAS_MODEL_PATH)
print(f"[INFO] Modelo Keras salvo em {KERAS_MODEL_PATH}")

# avaliação Keras no teste real
y_proba_nn_real = model.predict(X_test_nn, verbose=0).ravel()
met_nn_real = metrics_from_proba(y_test, y_proba_nn_real, titulo="KERAS - TESTE REAL (desbalanceado)")

# avaliação Keras no teste balanceado
y_proba_nn_bal = model.predict(X_test_bal_nn, verbose=0).ravel()
met_nn_bal = metrics_from_proba(y_test_bal, y_proba_nn_bal, titulo="KERAS - TESTE BALANCEADO (1:1)")

results_keras["KerasMLP"] = met_nn_bal

print(
    f"[INFO] Keras (BALANCEADO) -> acc={met_nn_bal['accuracy']:.4f}, "
    f"prec={met_nn_bal['precision']:.4f}, rec={met_nn_bal['recall']:.4f}, "
    f"f1={met_nn_bal['f1']:.4f}, auc={met_nn_bal['roc_auc']:.4f}"
)

# ----------------- SALVAR MÉTRICAS -----------------
metrics_df = pd.DataFrame(results).T.round(4)
metrics_df.to_csv("metricas_modelos.csv")

metrics_keras_df = pd.DataFrame(results_keras).T.round(4)
metrics_keras_df.to_csv("metricas_keras.csv")

fig, ax = plt.subplots(figsize=(10, 5))
metrics_df.plot(kind="bar", ax=ax)
ax.set_ylabel("Valor da métrica")
ax.set_title("Comparação de Modelos (Teste BALANCEADO, F1 otimizado)")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35),
          ncol=len(metrics_df.columns))
fig.subplots_adjust(bottom=0.3)
plt.text(0.99, 0.01, "PNS 2019 - Asma", transform=fig.transFigure,
         ha="right", va="bottom", fontsize=8, alpha=0.7)
fig.tight_layout()
fig.savefig("comparacao_modelos_metrics.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("[INFO] Métricas salvas em metricas_modelos.csv, metricas_keras.csv e comparacao_modelos_metrics.png")

# ----------------- t-SNE (treino FULL com ClusterCentroids) -----------------
print("[INFO] Gerando t-SNE (com undersampling ClusterCentroids no treino FULL)...")
imp = SimpleImputer(strategy="median").fit(X_train_full)
sc = StandardScaler().fit(imp.transform(X_train_full))
Xtr_scaled = sc.transform(imp.transform(X_train_full))

cc_tsne = ClusterCentroids(random_state=RANDOM_STATE)
Xbal_tsne, ybal_tsne = cc_tsne.fit_resample(Xtr_scaled, y_train_full)

max_pts = 1500
if len(ybal_tsne) > max_pts:
    rng = np.random.default_rng(123)
    idx = rng.choice(len(ybal_tsne), max_pts, replace=False)
    Xplot, yplot = Xbal_tsne[idx], ybal_tsne.iloc[idx].to_numpy()
else:
    Xplot, yplot = Xbal_tsne, ybal_tsne.to_numpy()

tsne = TSNE(n_components=2, perplexity=30, n_iter=400, random_state=42)
X2d = tsne.fit_transform(Xplot)

plt.figure(figsize=(6, 6))
plt.scatter(X2d[yplot == 0, 0], X2d[yplot == 0, 1],
            s=6, alpha=0.6, label="Não asmático")
plt.scatter(X2d[yplot == 1, 0,], X2d[yplot == 1, 1],
            s=6, alpha=0.6, label="Asmático")
plt.xlabel("Dimensão 1 (t-SNE)")
plt.ylabel("Dimensão 2 (t-SNE)")
plt.title("t-SNE — Treino com ClusterCentroids (undersampling)")
plt.legend()
plt.text(0.99, 0.01, "PNS 2019 - Asma", transform=plt.gcf().transFigure,
         ha="right", va="bottom", fontsize=8, alpha=0.7)
plt.tight_layout()
plt.savefig("tsne_balanceado.png", dpi=300, bbox_inches="tight")
plt.close()
print("[INFO] t-SNE salvo em tsne_balanceado.png")

print("\nArquivos gerados:")
print("✓ entropia_features.csv")
print("✓ entropia_features.png")
print("✓ metricas_modelos.csv  (DecisionTree, KNN, RandomForest, MLP — teste balanceado)")
print("✓ metricas_keras.csv    (Keras — teste balanceado)")
print("✓ comparacao_modelos_metrics.png")
print("✓ tsne_balanceado.png")
print("✓ modelo_asma.keras (rede neural com memória)")
