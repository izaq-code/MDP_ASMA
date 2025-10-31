# -*- coding: utf-8 -*-
# ===========================================================
# ANALISE PNS 2019 (BASE TRATADA) — DESCRIÇÃO CORRETA
# - Força tipo por lista do usuário (numéricas fixas; resto categóricas)
# - Detecta automaticamente a coluna de "Descrição" no dicionário
# - Lê CSV com auto-detecção de separador ("," ou ";") e trata BOM
# - Gera analise_resumo_completo.tex (modelo LaTeX validado)
# ===========================================================

import pandas as pd
import numpy as np
import math
from pathlib import Path

# ====================== CONFIG ======================
DICT_PATH = r"C:\Users\isaqu\OneDrive\Desktop\MDP_AVC\Data\dicionario_pns-tratado.xlsx"
DATA_PATH = r"C:\Users\isaqu\OneDrive\Desktop\MDP_AVC\Data\pns_2019_TRATADA.csv"
ENCODING  = "utf-8-sig"   # tenta com BOM; se precisar mude p/ None ou 'latin-1'
DECIMALS  = 6             # casas decimais nas métricas

# Se você souber EXATAMENTE o nome da coluna de descrição no dicionário, force aqui:
DESC_COL_OVERRIDE = "descrição variavel" # ex.: "Descrição", "Pergunta", "Descrição da variável"

# ======= LISTA FORÇADA PELO USUÁRIO =======
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
Q075
V0022
""".strip().split()
NUMERIC_FORCE = {v.strip().upper() for v in NUMERIC_FORCE_RAW if v.strip()}

# ====================== LEITURA ROBUSTA ======================
def read_pns(path, encoding=ENCODING):
    # 1) tenta vírgula
    try:
        df_ = pd.read_csv(path, sep=",", low_memory=False, dtype=str, encoding=encoding)
        if df_.shape[1] == 1 and ";" in df_.columns[0]:
            raise ValueError("detectado CSV com ';' em col única")
        return df_
    except Exception:
        pass
    # 2) tenta ponto e vírgula
    return pd.read_csv(path, sep=";", low_memory=False, dtype=str, encoding=encoding)

df = read_pns(DATA_PATH)
dic = pd.read_excel(DICT_PATH)

# Remove BOM e espaços
df.columns  = df.columns.str.replace('\ufeff', '', regex=True).str.strip()
dic.columns = pd.Index(dic.columns).str.replace('\ufeff', '', regex=True).str.strip()

# ====================== AUXILIARES ======================
def achar_coluna(candidatos, cols):
    lower_map = {c.lower(): c for c in cols}
    for nome in candidatos:
        k = nome.lower().strip()
        if k in lower_map:
            return lower_map[k]
    return None

def series_from_column(df_: pd.DataFrame, colname: str) -> pd.Series:
    obj = df_.loc[:, colname]
    return obj.iloc[:, 0] if isinstance(obj, pd.DataFrame) else obj

def tex_escape(s: str) -> str:
    if s is None: return ""
    s = str(s).replace("\r", " ").replace("\n", " ")
    for a, b in [("\\","\\textbackslash{}"),("{","\\{"),("}","\\}"),
                 ("$","\\$"),("&","\\&"),("#","\\#"),("%","\\%"),
                 ("_","\\_"),("~","\\textasciitilde{}"),("^","\\textasciicircum{}")]:
        s = s.replace(a,b)
    return " ".join(s.split())

def fmt_num(x, nd=DECIMALS):
    if x is None: return ""
    try:
        x = float(x)
        if math.isnan(x): return ""
        return f"{x:.{nd}f}"
    except Exception:
        return ""

# ====================== DICIONÁRIO PADRÃO (ROBUSTO) ======================
# CÓDIGO (obrigatório)
col_codigo = achar_coluna(
    ["código da variável","codigo da variavel","código","codigo","variavel","variable","var","nome","coluna"],
    dic.columns
)
if col_codigo is None:
    raise ValueError("Não encontrei a coluna de código no dicionário (ex.: 'Código da variável').")

# Escolha robusta da coluna de DESCRIÇÃO
def escolher_coluna_descricao(dic_df: pd.DataFrame, col_codigo: str) -> str:
    if DESC_COL_OVERRIDE and DESC_COL_OVERRIDE in dic_df.columns:
        print(f"[Diag] Coluna de descrição FORÇADA: {DESC_COL_OVERRIDE}")
        return DESC_COL_OVERRIDE

    candidatos_nominais = [
        "descrição da variável","descricao da variavel","pergunta completa","enunciado","pergunta",
        "descrição","descricao","rótulo","rotulo","label","texto","título","titulo",
        "descrição reduzida","descricao reduzida","descrição curta","descricao curta"
    ]
    col_desc_nominal = achar_coluna(candidatos_nominais, dic_df.columns)
    if col_desc_nominal:
        print(f"[Diag] Coluna de descrição por nome: {col_desc_nominal}")
        return col_desc_nominal

    # Heurística de conteúdo (texto longo/diverso, pouco repetido)
    scores = []
    n = len(dic_df)
    for c in dic_df.columns:
        if c == col_codigo: 
            continue
        s = series_from_column(dic_df, c).astype(str)
        # ignora colunas muito curtas ou numéricas/IDs
        if s.str.len().mean() < 4:
            continue
        sample = s.head(200).str.replace(r"\s+", " ", regex=True).str.strip()
        if sample.str.fullmatch(r"[0-9\.\-,;:/\\ ]{1,}$", na=False).mean() > 0.7:
            continue
        mean_len = s.str.len().mean()
        nuniq = s.nunique(dropna=True)
        top_freq = (s.value_counts(dropna=True).iloc[0]/n) if n>0 and not s.value_counts(dropna=True).empty else 0.0
        diversity = (nuniq / max(n,1))
        score = (mean_len * 0.7) + (diversity * 100.0 * 0.3) - (top_freq * 10.0)
        scores.append((c, mean_len, diversity, top_freq, score))
    if not scores:
        # fallback: primeira não-código
        fallback = [c for c in dic_df.columns if c != col_codigo][0]
        print(f"[Diag] Heurística sem candidatos; usando fallback: {fallback}")
        return fallback
    scores_sorted = sorted(scores, key=lambda x: x[-1], reverse=True)
    best = scores_sorted[0][0]
    print("[Diag] Ranking possíveis colunas de descrição (top 5):")
    for c, ml, div, tf, sc in scores_sorted[:5]:
        print(f"  - {c} | mean_len={ml:.1f}, diversity={div:.3f}, top_freq={tf:.3f}, score={sc:.1f}")
    print(f"[Diag] Coluna de descrição escolhida: {best}")
    return best

col_desc = escolher_coluna_descricao(dic, col_codigo)

# Colunas opcionais
col_tipo  = achar_coluna(["tipo","natureza","tipo da variável","tipo da variavel","formato","classe"], dic.columns)
col_class = achar_coluna(["classificação","classificacao","escala","nível de medida","nivel de medida","nivel","ordem"], dic.columns)

# Monta dicionário padronizado
dic_std = pd.DataFrame({"codigo": series_from_column(dic, col_codigo).astype(str).str.strip()})
dic_std["tipo_raw"]  = series_from_column(dic, col_tipo).astype(str).str.strip().str.lower()   if col_tipo  else ""
dic_std["class_raw"] = series_from_column(dic, col_class).astype(str).str.strip().str.lower()  if col_class else ""
dic_std["descricao"] = series_from_column(dic, col_desc).astype(str).fillna("").str.strip()

# Limpa e cruza com dataset
dic_std = dic_std[dic_std["codigo"].ne("")].drop_duplicates(subset=["codigo"])
dic_std = dic_std[dic_std["codigo"].isin(df.columns)].reset_index(drop=True)

print(f"[Diag] Coluna de CÓDIGO: {col_codigo}")
print(f"[Diag] Coluna de DESCRIÇÃO: {col_desc}")
print("[Diag] Amostras:")
print(dic_std[["codigo","descricao"]].head(5))

# ====================== TIPOS: NUMÉRICAS x CATEGÓRICAS ======================
# Mapa UPPER->real
df_cols_upper = {c.upper(): c for c in df.columns}

# Normaliza valores: numéricas convertidas, categóricas ficam string e tratam "falsos NA"
for c_up, c_real in df_cols_upper.items():
    if c_up in NUMERIC_FORCE:
        df[c_real] = pd.to_numeric(df[c_real].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    else:
        df[c_real] = df[c_real].astype(str).str.strip()
        df.loc[df[c_real].isin({"", " ", "NA", "N/A", "nan", "NaN", "None", "NULL", "Null", "null"}), c_real] = np.nan

# Seleção final
vars_num = [df_cols_upper[v] for v in NUMERIC_FORCE if v in df_cols_upper]
vars_cat = [c for c in dic_std["codigo"].tolist() if c not in set(vars_num)]

print(f"[Diag] Variáveis no dicionário encontradas no dataset: {dic_std.shape[0]}")
print(f"[Diag] Numéricas (forçadas): {len(vars_num)} | Categóricas (restante): {len(vars_cat)}")

# ====================== ESTATÍSTICAS ======================
res_num = []
for col in sorted(vars_num):
    s = pd.to_numeric(df[col], errors="coerce")
    n_total = int(s.shape[0]); n_missing = int(s.isna().sum())
    s_valid = s.dropna()
    if s_valid.empty:
        res_num.append({
            "codigo": col, "n": n_total, "ausentes": n_missing,
            "min": np.nan, "max": np.nan, "media": np.nan,
            "mediana": np.nan, "desvio_padrao": np.nan
        })
    else:
        res_num.append({
            "codigo": col, "n": n_total, "ausentes": n_missing,
            "min": float(s_valid.min()), "max": float(s_valid.max()),
            "media": float(s_valid.mean()), "mediana": float(s_valid.median()),
            "desvio_padrao": float(s_valid.std(ddof=1))
        })

res_cat = []
for col in sorted(vars_cat):
    s = df[col]
    n_total = int(s.shape[0]); n_missing = int(s.isna().sum())
    vc = s.value_counts(dropna=True)
    moda = vc.index[0] if not vc.empty else np.nan
    res_cat.append({
        "codigo": col, "n": n_total, "ausentes": n_missing,
        "n_categorias": int(vc.shape[0]),
        "moda": moda,
        "tipo_categorica": "desconhecida"
    })

res_num_df = pd.DataFrame(res_num) if res_num else pd.DataFrame(columns=[
    "codigo","n","ausentes","min","max","media","mediana","desvio_padrao"
])
res_cat_df = pd.DataFrame(res_cat) if res_cat else pd.DataFrame(columns=[
    "codigo","n","ausentes","n_categorias","moda","tipo_categorica"
])

# ====================== DESCRIÇÃO ======================
desc_map = dict(zip(dic_std["codigo"], dic_std["descricao"]))
if not res_num_df.empty: res_num_df["descricao"] = res_num_df["codigo"].map(desc_map).fillna("")
if not res_cat_df.empty: res_cat_df["descricao"] = res_cat_df["codigo"].map(desc_map).fillna("")

# ====================== LINHAS TEX ======================
def to_tex_rows_num(df_rows):
    lines = []
    for _, r in df_rows.iterrows():
        line = (
            f"{tex_escape(r['codigo'])} & "
            f"{tex_escape(str(r['descricao']))} & "
            f"{int(r['n'])} & {int(r['ausentes'])} & "
            f"{fmt_num(r['min'])} & {fmt_num(r['max'])} & "
            f"{fmt_num(r['media'])} & {fmt_num(r['mediana'])} & "
            f"{fmt_num(r['desvio_padrao'])} \\\\"
        )
        lines.append(line)
    return "\n".join(lines)

def to_tex_rows_cat(df_rows):
    lines = []
    for _, r in df_rows.iterrows():
        line = (
            f"{tex_escape(r['codigo'])} & "
            f"{tex_escape(str(r['descricao']))} & "
            f"{int(r['n'])} & {int(r['ausentes'])} & "
            f"{int(r['n_categorias'])} & {tex_escape(str(r['moda']))} & "
            f"{tex_escape(str(r['tipo_categorica']))} \\\\"
        )
        lines.append(line)
    return "\n".join(lines)

numeric_rows = to_tex_rows_num(res_num_df.sort_values("codigo")) if not res_num_df.empty else ""
categorical_rows = to_tex_rows_cat(res_cat_df.sort_values("codigo")) if not res_cat_df.empty else ""

# ====================== MODELO TEX ======================
doc_template = r"""
\documentclass[a4paper,12pt]{article}

% ===================== PACOTES =====================
\usepackage[utf8]{inputenc}
\usepackage[brazil]{babel}
\usepackage{geometry}
\geometry{margin=2.5cm}

\usepackage{booktabs,longtable,array}
\usepackage{siunitx}
\usepackage[table]{xcolor}
\usepackage{caption}

% ===================== AJUSTES GERAIS =====================
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}

\setlength{\LTleft}{0pt}
\setlength{\LTright}{0pt}
\setlength{\LTpre}{8pt}
\setlength{\LTpost}{8pt}

\setlength{\tabcolsep}{6pt}
\renewcommand{\arraystretch}{1.25}

\captionsetup{skip=8pt,width=\linewidth}

\sisetup{
  round-mode=places,
  round-precision=2,
  detect-weight=true,
  detect-inline-weight=math,
  group-separator = {.},
  output-decimal-marker = {,},
  table-number-alignment = center,
  table-text-font = \tiny
}

\begin{document}

\section*{Resumo das Variáveis — PNS 2019 (Base Tratada)}

A seguir são apresentados os resumos estatísticos das variáveis do conjunto de dados da Pesquisa Nacional de Saúde (PNS 2019), separados entre variáveis numéricas e categóricas.  
As estatísticas incluem medidas de posição (média, mediana), dispersão (desvio-padrão) e informações de completude (ausentes).

% ===================== RESUMO NUMÉRICAS =====================
\begingroup\tiny
\rowcolors{3}{black!2}{white}
\begin{longtable}{@{}
    l
    L{5.8cm}
    S[table-format=6.0]
    S[table-format=6.0]
    S[table-format=3.0]
    S[table-format=3.0]
    S[table-format=3.1]
    S[table-format=3.1]
    S[table-format=3.1]
@{}}
\caption{Resumo das variáveis numéricas}\label{tab:resumo_numericas}\\
\toprule
\textbf{Código} & \textbf{Descrição} & \textbf{n} & \textbf{Ausentes} & \textbf{Mín} & \textbf{Máx} & \textbf{Média} & \textbf{Mediana} & \textbf{DP} \\
\midrule
\endfirsthead
\caption[]{Resumo das variáveis numéricas (continuação)}\\
\toprule
\textbf{Código} & \textbf{Descrição} & \textbf{n} & \textbf{Ausentes} & \textbf{Mín} & \textbf{Máx} & \textbf{Média} & \textbf{Mediana} & \textbf{DP} \\
\midrule
\endhead
\midrule
\multicolumn{9}{r}{\emph{Continua na próxima página}}\\
\midrule
\endfoot
\bottomrule
\endlastfoot
__NUMERIC_ROWS__
\end{longtable}
\endgroup

% ===================== RESUMO CATEGÓRICAS =====================
\begingroup\tiny
\rowcolors{3}{black!2}{white}
\begin{longtable}{@{}
    l
    L{6.2cm}
    S[table-format=6.0]
    S[table-format=6.0]
    S[table-format=2.0]
    S[table-format=1.0]
    l
@{}}
\caption{Resumo das variáveis categóricas}\label{tab:resumo_categoricas}\\
\toprule
\textbf{Código} & \textbf{Descrição} & \textbf{n} & \textbf{Ausentes} & \textbf{\#Categorias} & \textbf{Moda} & \textbf{Tipo} \\
\midrule
\endfirsthead
\caption[]{Resumo das variáveis categóricas (continuação)}\\
\toprule
\textbf{Código} & \textbf{Descrição} & \textbf{n} & \textbf{Ausentes} & \textbf{\#Categorias} & \textbf{Moda} & \textbf{Tipo} \\
\midrule
\endhead
\midrule
\multicolumn{7}{r}{\emph{Continua na próxima página}}\\
\midrule
\endfoot
\bottomrule
\endlastfoot
__CATEGORICAL_ROWS__
\end{longtable}
\endgroup

\end{document}
"""

# ====================== GRAVA ======================
doc_final = (doc_template
             .replace("__NUMERIC_ROWS__", numeric_rows)
             .replace("__CATEGORICAL_ROWS__", categorical_rows))

Path("analise_resumo_completo.tex").write_text(doc_final, encoding="utf-8")
print("✅ OK: gerado 'analise_resumo_completo.tex'. Compile com pdflatex/xelatex/lualatex.")
