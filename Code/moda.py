import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# -----------------------------
# 1) Leitura dos dados
# -----------------------------
arquivo = "./Data/pns_2019.csv"
df = pd.read_csv(arquivo, sep=";", low_memory=False)

# -----------------------------
# 2) Limpeza rápida das colunas
# -----------------------------
df.columns = df.columns.str.strip().str.replace(" ", "").str.upper()

# -----------------------------
# 3) Listas de variáveis ajustadas
# -----------------------------
variaveis_numericas = ["Q075", "P01601", "P018", "P019", "P023", "J12", "J38"]
variaveis_categoricas = ["Q074", "Q076", "Q07601", "Q078", "P068", "P050",
                         "I1B", "J11A", "Q07705", "Q07706", "Q07709", "Q07710"]

# -----------------------------
# 4) Estatísticas variáveis numéricas
# -----------------------------
estat_num = []
for var in variaveis_numericas:
    if var in df.columns:
        serie = df[var].dropna()
        estat_num.append({
            "Variável": var,
            "Min": serie.min(),
            "Max": serie.max(),
            "Média": serie.mean(),
            "Mediana": serie.median(),
            "Moda": serie.mode()[0] if not serie.mode().empty else None,
            "Desvio_Padrão": serie.std(),
            "Dados_Ausentes": df[var].isna().sum()
        })

estat_num_df = pd.DataFrame(estat_num)

# -----------------------------
# 5) Estatísticas variáveis categóricas
# -----------------------------
estat_cat = []
for var in variaveis_categoricas:
    if var in df.columns:
        serie = df[var].dropna()
        freq = serie.value_counts().to_dict()
        estat_cat.append({
            "Variável": var,
            "Tipo": "Ordinal" if var in ["Q078", "P068", "J11A"] else "Nominal/Dicótoma",
            "Moda": serie.mode()[0] if not serie.mode().empty else None,
            "Dados_Ausentes": df[var].isna().sum(),
            "Frequência": freq
        })

estat_cat_df = pd.DataFrame(estat_cat)

# -----------------------------
# 6) Função para salvar DataFrame como PNG
# -----------------------------
def salvar_tabela_png(df, nome_arquivo):
    fig, ax = plt.subplots(figsize=(12, 0.5*len(df)+1))
    ax.axis("tight")
    ax.axis("off")
    tabela = ax.table(cellText=df.values, colLabels=df.columns, loc="center")

    # Ajuste visual
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(8)
    tabela.scale(1.2, 1.2)

    plt.savefig(nome_arquivo, bbox_inches="tight", dpi=300)
    plt.close()

# -----------------------------
# 7) Salvando as duas tabelas em PNG
# -----------------------------
salvar_tabela_png(estat_num_df, "estatisticas_numericas.png")
salvar_tabela_png(estat_cat_df.drop(columns=["Frequência"]), "estatisticas_categoricas.png")  # drop só para caber melhor

print("✅ Imagens geradas: estatisticas_numericas.png e estatisticas_categoricas.png")
