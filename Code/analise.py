# -*- coding: utf-8 -*-
# ===========================================================
# ANALISE PNS 2019 - FORÇA TIPO PELO USUÁRIO (NUMÉRICAS FIXAS)
# Gera: analise_resumo_completo.tex (modelo LaTeX validado)
# ===========================================================

import pandas as pd
import numpy as np
import math
from pathlib import Path

# ====================== CONFIG ======================
DICT_PATH = "./Data/dicionario_pns-tratado.xlsx"
DATA_PATH = "./Data/pns_2019.csv"
CSV_SEP   = ";"
ENCODING  = None
DECIMALS  = 6  # casas decimais para métricas

# ======= LISTA FORÇADA PELO USUÁRIO =======
# Todas as variáveis listadas aqui serão NUMÉRICAS; o resto será CATEGÓRICA.
NUMERIC_FORCE_RAW = """
A02305
A02306
A02307
A02401
A02402
C001
C008
P00103
P00104
P00403
P00404
P006
P00901
P01001
P01101
P013
P015
P02001
P02101
P01601
P018
P019
P02002
P023
P02501
P02602
P02801
P029
P03202
P035
P03701
P03702
P03904
P03905
P03906
P04001
P04101
P04102
P042
P04301
P04302
P044
P04401
P04405
P04406
P053
P05402
P05403
P05405
P05406
P05408
P05409
P05411
P05412
P05414
P05415
P05417
P05418
P05421
P05422
P05601
P05602
P05603
P05604
P05605
P057
P05801
P05802
P05901
P05902
P05903
P05904
Q075
""".strip().split()

NUMERIC_FORCE = {v.strip().upper() for v in NUMERIC_FORCE_RAW if v.strip()}

# ====================== LEITURA ======================
dic = pd.read_excel(DICT_PATH)
dic.columns = [c.strip().lower() for c in dic.columns]
df = pd.read_csv(DATA_PATH, sep=CSV_SEP, low_memory=False, dtype=str, encoding=ENCODING)

# ====================== AUXILIARES ======================
def achar_coluna(candidatos, cols):
    lower_map = {c.lower(): c for c in cols}
    for nome in candidatos:
        k = nome.lower().strip()
        if k in lower_map:
            return lower_map[k]
    return None

def series_from_column(df_: pd.DataFrame, colname: str) -> pd.Series:
    """Garante Series mesmo que haja colunas repetidas no Excel."""
    obj = df_.loc[:, colname]
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0]
    return obj

def tex_escape(s: str) -> str:
    if s is None: return ""
    s = str(s).replace("\r", " ").replace("\n", " ")
    for a, b in [("\\","\\textbackslash{}"),("{","\\{"),("}","\\}"),
                 ("$","\\$"),("&","\\&"),("#","\\#"),("%","\\%"),
                 ("_","\\_"),("~","\\textasciitilde{}"),("^","\\textasciicircum{}")]:
        s = s.replace(a,b)
    return " ".join(s.split())

def fmt_num(x, nd=DECIMALS):
    if x is None or (isinstance(x, float) and math.isnan(x)) or (isinstance(x, str) and x.strip()==""):
        return ""
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return ""

# ====================== DICIONÁRIO PADRÃO ======================
col_codigo = achar_coluna(
    ["código da variável","codigo da variavel","codigo","variavel","nome","coluna","variable","var"],
    dic.columns
)
col_tipo = achar_coluna(["tipo","natureza","tipo da variável","tipo da variavel","formato","classe"], dic.columns)
col_class = achar_coluna(["classificação","classificacao","escala","nível de medida","nivel de medida","nivel","ordem"], dic.columns)
col_desc = achar_coluna(["descrição","descricao","pergunta","label","rótulo","rotulo","titulo","título"], dic.columns)

if col_codigo is None:
    raise ValueError("Não encontrei a coluna de código no dicionário (ex.: 'Código da variável').")

dic_std = pd.DataFrame({"codigo": series_from_column(dic, col_codigo).astype(str).str.strip()})
dic_std["tipo_raw"]  = series_from_column(dic, col_tipo).astype(str).str.strip().str.lower()   if col_tipo  else ""
dic_std["class_raw"] = series_from_column(dic, col_class).astype(str).str.strip().str.lower()  if col_class else ""
dic_std["descricao"] = series_from_column(dic, col_desc).astype(str).fillna("").str.strip()    if col_desc  else ""

# mantém somente códigos existentes no dataset
dic_std = dic_std[dic_std["codigo"].ne("")].drop_duplicates(subset=["codigo"])
dic_std = dic_std[dic_std["codigo"].isin(df.columns)].reset_index(drop=True)

# Normaliza caixa dos nomes das colunas do dataset para bater com NUMERIC_FORCE
# (mas SEM renomear o df real; só usamos para checagem)
df_cols_upper = {c.upper(): c for c in df.columns}

# ====================== CLASSIFICAÇÃO (FORÇADA) ======================
# Numéricas = exatamente as da lista fornecida que existirem no dataset
vars_num = []
for code_up in NUMERIC_FORCE:
    if code_up in df_cols_upper:
        vars_num.append(df_cols_upper[code_up])

# Categóricas = todas as demais presentes no dataset & no dicionário
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
        "tipo_categorica": "desconhecida"  # sem inferência: tudo que não está na lista é categórica
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
numeric_rows = ""
if not res_num_df.empty:
    res_num_df = res_num_df.sort_values("codigo")
    lines = []
    for _, r in res_num_df.iterrows():
        line = (
            f"{tex_escape(r['codigo'])} & "
            f"{tex_escape(str(r['descricao']))} & "
            f"{int(r['n'])} & {int(r['ausentes'])} & "
            f"{fmt_num(r['min'])} & {fmt_num(r['max'])} & "
            f"{fmt_num(r['media'])} & {fmt_num(r['mediana'])} & "
            f"{fmt_num(r['desvio_padrao'])} \\\\"
        )
        lines.append(line)
    numeric_rows = "\n".join(lines)

categorical_rows = ""
if not res_cat_df.empty:
    res_cat_df = res_cat_df.sort_values("codigo")
    lines = []
    for _, r in res_cat_df.iterrows():
        line = (
            f"{tex_escape(r['codigo'])} & "
            f"{tex_escape(str(r['descricao']))} & "
            f"{int(r['n'])} & {int(r['ausentes'])} & "
            f"{int(r['n_categorias'])} & {tex_escape(str(r['moda']))} & "
            f"{tex_escape(str(r['tipo_categorica']))} \\\\"
        )
        lines.append(line)
    categorical_rows = "\n".join(lines)

# ====================== MODELO TEX (VALIDADO) ======================
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

% Longtable e espaçamentos
\setlength{\LTleft}{0pt}
\setlength{\LTright}{0pt}
\setlength{\LTpre}{8pt}
\setlength{\LTpost}{8pt}

% 🔹 Aumentei mais o espaço entre colunas e linhas
\setlength{\tabcolsep}{6pt}       % antes 4pt → agora mais espaçado
\renewcommand{\arraystretch}{1.25} % linhas mais altas

% Legendas
\captionsetup{skip=8pt,width=\linewidth}

% Configuração de números (mantém letra pequena e alinhada)
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

% ===================== DOCUMENTO =====================
\begin{document}

\section*{Resumo das Variáveis — PNS 2019}

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
