# -*- coding: utf-8 -*-
"""
Tratamento PNS 2019 com dependências:
- CSV da PNS em ; (ponto e vírgula) e possivelmente com BOM
- 0 = "não se aplica" quando a base não habilita a dependente
- Imputação média (num) / moda (cat) quando deveria responder e está ausente
- Não dependentes: média/moda global
- Detecção automática do(s) valor(es) habilitadores
- Relatórios de imputação e missings
"""

import os
import pandas as pd
import numpy as np

# ======================== CONFIG ========================

BASE_PATH = r"C:\Users\isaqu\OneDrive\Desktop\MDP_AVC\Data\pns_2019.csv"
DIC_PATH  = r"C:\Users\isaqu\OneDrive\Desktop\MDP_AVC\Data\dicionario_pns-tratado.xlsx"

NUMERIC_FORCE_RAW = """
A02305
A02306
A02307
C001
C008
P00104
P00404
P006
P00901
P01001
P01101
P013
P015
P02001
P01601
P018
P019
P02002
P023
P02501
P02602
P053
P05901
P05902
P05903
P05904
Q075
"""
NUMERIC_VARS = set([x.strip() for x in NUMERIC_FORCE_RAW.strip().splitlines() if x.strip()])

COL_VAR       = "Código da variável"
COL_DEP_FLAG  = "Dependente"
COL_DEP_CODE  = "Código da variável dependente"

ENABLER_MIN_SHARE = 0.60
ENABLER_MIN_COUNT = 50

ENABLER_EXCEPTIONS = {
    # "Q074": {"1"},
}

MISSING_LIKE = {"", " ", "NA", "N/A", "nan", "NaN", "None", "NULL", "Null", "null"}

# ======================== HELPERS ========================

def is_missing_series(s: pd.Series) -> pd.Series:
    s_str = s.astype(str)
    return s.isna() | s_str.isin(MISSING_LIKE)

def safe_mode(series: pd.Series):
    x = series.dropna()
    m = x.mode()
    return (m.iloc[0] if not m.empty else np.nan)

def impute_numeric_inplace(s: pd.Series, value):
    mask = s.isna()
    s.loc[mask] = value
    return int(mask.sum())

def impute_categorical_inplace(s: pd.Series, value):
    mask = is_missing_series(s)
    s.loc[mask] = value
    return int(mask.sum())

def detect_enablers(df: pd.DataFrame, var_base: str, var_dep: str,
                    min_share=ENABLER_MIN_SHARE, min_count=ENABLER_MIN_COUNT) -> set:
    if var_base not in df.columns or var_dep not in df.columns:
        return set()
    base_vals = df[var_base].astype(str)
    stats = []
    for val, sub in df.groupby(base_vals, dropna=False):
        n = len(sub)
        if n == 0:
            continue
        share = (~is_missing_series(sub[var_dep])).mean()
        stats.append((str(val), n, float(share)))
    if not stats:
        return set()
    enablers = {val for (val, n, share) in stats if (share >= min_share and n >= min_count)}
    if enablers:
        return enablers
    max_share = max(share for (_, _, share) in stats)
    top = {val for (val, _, share) in stats if share == max_share}
    return top

def normalize_columns(df: pd.DataFrame):
    df.columns = df.columns.str.strip()
    # Remove BOM se col 0 vier com 'ï»¿V0001'
    df.columns = df.columns.str.replace('\ufeff', '', regex=True)
    return df

def normalize_values(df: pd.DataFrame):
    """
    - Converte variáveis da lista NUMERIC_VARS para numéricas
    - Para categóricas: tira espaços, substitui MISSING_LIKE por NaN
    - Se algum numérico vier com vírgula decimal, troca para ponto antes do to_numeric
    """
    for col in df.columns:
        if col in NUMERIC_VARS:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = df[col].astype(str).str.strip()
            df.loc[is_missing_series(df[col]), col] = np.nan
    return df

def robust_dep_flag(s: pd.Series) -> pd.Series:
    """
    Interpreta 'Dependente' como verdadeiro se for {1, '1', '1.0', 'true', 'True', 'sim', 'Sim'}.
    """
    truthy = {"1", "1.0", "true", "True", "sim", "Sim", "TRUE", "SIM"}
    return s.astype(str).str.strip().isin(truthy)

# ======================== CORE ========================

def treat_dependents(df: pd.DataFrame, dic: pd.DataFrame, stats: dict) -> pd.DataFrame:
    dep_map = dic[robust_dep_flag(dic[COL_DEP_FLAG])][[COL_VAR, COL_DEP_CODE]].dropna()
    dep_map[COL_VAR] = dep_map[COL_VAR].astype(str).str.strip()
    dep_map[COL_DEP_CODE] = dep_map[COL_DEP_CODE].astype(str).str.strip()

    for var_dep, var_base in dep_map.itertuples(index=False, name=None):
        if var_dep not in df.columns or var_base not in df.columns:
            continue

        if var_base in ENABLER_EXCEPTIONS and ENABLER_EXCEPTIONS[var_base]:
            enablers = set(map(str, ENABLER_EXCEPTIONS[var_base]))
        else:
            enablers = detect_enablers(df, var_base, var_dep)
        if not enablers:
            enablers = {"1"}  # fallback final

        base_str = df[var_base].astype(str)
        cond_deveria = base_str.isin(enablers)
        cond_nao_deveria = ~cond_deveria

        stats.setdefault(var_dep, {"nao_aplica_zero": 0, "imput_media": 0, "imput_moda": 0,
                                   "tipo": "num" if var_dep in NUMERIC_VARS else "cat"})

        # Zera onde não se aplica
        if var_dep in NUMERIC_VARS:
            df[var_dep] = pd.to_numeric(df[var_dep], errors='coerce')
            df.loc[cond_nao_deveria, var_dep] = 0.0
            stats[var_dep]["nao_aplica_zero"] += int(cond_nao_deveria.sum())
        else:
            df[var_dep] = df[var_dep].astype(str).str.strip()
            df.loc[is_missing_series(df[var_dep]), var_dep] = np.nan
            df.loc[cond_nao_deveria, var_dep] = "0"
            stats[var_dep]["nao_aplica_zero"] += int(cond_nao_deveria.sum())

        # Imputa onde deveria e está ausente
        if var_dep in NUMERIC_VARS:
            grupo = df.loc[cond_deveria, var_dep]
            mean_val = grupo.mean(skipna=True)
            if pd.isna(mean_val):
                mean_val = df[var_dep].mean(skipna=True)
                if pd.isna(mean_val):
                    mean_val = 0.0
            imputados = impute_numeric_inplace(df.loc[cond_deveria, var_dep], mean_val)
            stats[var_dep]["imput_media"] += imputados
        else:
            grupo = df.loc[cond_deveria, var_dep]
            mode_val = safe_mode(grupo)
            if pd.isna(mode_val):
                mode_val = safe_mode(df[var_dep])
                if pd.isna(mode_val):
                    mode_val = "1"
            imputados = impute_categorical_inplace(df.loc[cond_deveria, var_dep], mode_val)
            stats[var_dep]["imput_moda"] += imputados

    return df

def treat_non_dependents(df: pd.DataFrame, dic: pd.DataFrame, stats: dict) -> pd.DataFrame:
    nao_dep = dic[~robust_dep_flag(dic[COL_DEP_FLAG])][COL_VAR].astype(str).str.strip()
    nao_dep = [c for c in nao_dep if c in df.columns]

    for var in nao_dep:
        stats.setdefault(var, {"nao_aplica_zero": 0, "imput_media": 0, "imput_moda": 0,
                               "tipo": "num" if var in NUMERIC_VARS else "cat"})
        if var in NUMERIC_VARS:
            df[var] = pd.to_numeric(df[var], errors='coerce')
            mean_val = df[var].mean(skipna=True)
            if pd.isna(mean_val):
                mean_val = 0.0
            imputados = impute_numeric_inplace(df[var], mean_val)
            stats[var]["imput_media"] += imputados
        else:
            df[var] = df[var].astype(str).str.strip()
            df.loc[is_missing_series(df[var]), var] = np.nan
            mode_val = safe_mode(df[var])
            if pd.isna(mode_val):
                mode_val = "1"
            imputados = impute_categorical_inplace(df[var], mode_val)
            stats[var]["imput_moda"] += imputados
    return df

def missing_report(df: pd.DataFrame, top=30) -> pd.Series:
    miss = df.isna().mean().sort_values(ascending=False)
    return miss.head(top)

# ======================== MAIN ========================

def main():
    # ---------- Carregar corretamente a PNS ----------
    print("[INFO] Carregando base e dicionário...")
    # CSV da PNS costuma vir com ; e BOM
    try:
        df = pd.read_csv(BASE_PATH, sep=';', encoding='utf-8-sig', dtype='object', engine='python')
    except Exception:
        df = pd.read_csv(BASE_PATH, sep=';', encoding='latin-1', dtype='object', engine='python')

    dic = pd.read_excel(DIC_PATH)

    # ---------- Normalizar ----------
    df  = normalize_columns(df)
    dic = normalize_columns(dic)

    if not {COL_VAR, COL_DEP_FLAG, COL_DEP_CODE}.issubset(dic.columns):
        raise ValueError(f"Dicionário não contém colunas obrigatórias: {COL_VAR}, {COL_DEP_FLAG}, {COL_DEP_CODE}")

    df = normalize_values(df)

    # ---------- Checks rápidos ----------
    print(f"[INFO] Linhas x Colunas: {df.shape[0]} x {df.shape[1]}")
    print(f"[INFO] Primeiras colunas: {list(df.columns[:12])}")

    # ---------- Tratar ----------
    stats = {}
    print("[INFO] Tratando variáveis dependentes (detecção automática de habilitadores)...")
    df = treat_dependents(df, dic, stats)

    print("[INFO] Tratando variáveis não dependentes (média/moda)...")
    df = treat_non_dependents(df, dic, stats)

    # ---------- Relatórios ----------
    print("\n===== RESUMO DE IMPUTAÇÃO =====")
    total_zero = total_media = total_moda = 0
    linhas = []
    for var, st in sorted(stats.items()):
        total_zero  += st["nao_aplica_zero"]
        total_media += st["imput_media"]
        total_moda  += st["imput_moda"]
        linhas.append([var, st["tipo"], st["nao_aplica_zero"], st["imput_media"], st["imput_moda"]])

    if linhas:
        resumo = pd.DataFrame(linhas, columns=["variavel", "tipo", "nao_aplica_zero", "imput_media", "imput_moda"])
        print(resumo.sort_values(["nao_aplica_zero","imput_media","imput_moda"], ascending=False).head(25))
    else:
        resumo = pd.DataFrame(columns=["variavel","tipo","nao_aplica_zero","imput_media","imput_moda"])
        print("(vazio) — verifique se o dicionário tem 'Dependente' preenchido com 1/sim/true e se os códigos batem com as colunas da PNS.")

    print("\n[TOTAIS]")
    print(f"  0 (não se aplica): {total_zero:,}")
    print(f"  Imputação média : {total_media:,}")
    print(f"  Imputação moda  : {total_moda:,}")

    print("\n===== TOP COLUNAS COM MAIS NA APÓS TRATAMENTO =====")
    print(missing_report(df, top=30))

    # ---------- Salvar ----------
    out_csv = BASE_PATH.replace(".csv", "_TRATADA.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[OK] Base tratada salva em: {out_csv}")

    try:
        out_parquet = BASE_PATH.replace(".csv", "_TRATADA.parquet")
        df.to_parquet(out_parquet, index=False)
        print(f"[OK] Versão Parquet salva em: {out_parquet}")
    except Exception as e:
        print(f"[WARN] Parquet opcional não salvo: {e}")

    return df, resumo

if __name__ == "__main__":
    df_tratada, resumo_stats = main()
