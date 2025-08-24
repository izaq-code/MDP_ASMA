import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# -----------------------------
# 1) Leitura dos dados
# -----------------------------
arquivo = "./Data/pns_2019.csv"   # ou .xlsx
df = pd.read_csv(arquivo, sep=";", low_memory=False)
# df = pd.read_excel(arquivo)  # se for excel

# -----------------------------
# 2) Filtrar diagnóstico
# -----------------------------
df_filtrado = df[df["Q068"] == 1]

# -----------------------------
# 3) Mapeamento UF -> Nome + Região
# -----------------------------
ufs = {
    11: ("Rondônia", "Norte"),
    12: ("Acre", "Norte"),
    13: ("Amazonas", "Norte"),
    14: ("Roraima", "Norte"),
    15: ("Pará", "Norte"),
    16: ("Amapá", "Norte"),
    17: ("Tocantins", "Norte"),
    21: ("Maranhão", "Nordeste"),
    22: ("Piauí", "Nordeste"),
    23: ("Ceará", "Nordeste"),
    24: ("Rio Grande do Norte", "Nordeste"),
    25: ("Paraíba", "Nordeste"),
    26: ("Pernambuco", "Nordeste"),
    27: ("Alagoas", "Nordeste"),
    28: ("Sergipe", "Nordeste"),
    29: ("Bahia", "Nordeste"),
    31: ("Minas Gerais", "Sudeste"),
    32: ("Espírito Santo", "Sudeste"),
    33: ("Rio de Janeiro", "Sudeste"),
    35: ("São Paulo", "Sudeste"),
    41: ("Paraná", "Sul"),
    42: ("Santa Catarina", "Sul"),
    43: ("Rio Grande do Sul", "Sul"),
    50: ("Mato Grosso do Sul", "Centro-Oeste"),
    51: ("Mato Grosso", "Centro-Oeste"),
    52: ("Goiás", "Centro-Oeste"),
    53: ("Distrito Federal", "Centro-Oeste"),
}

# Criar colunas novas no DataFrame
df_filtrado["UF_nome"] = df_filtrado["V0001"].map(lambda x: ufs.get(x, ("Desconhecido", "Outro"))[0])
df_filtrado["Regiao"] = df_filtrado["V0001"].map(lambda x: ufs.get(x, ("Desconhecido", "Outro"))[1])

# -----------------------------
# 4) Contagem por UF e Região (absoluta e relativa)
# -----------------------------
total = len(df_filtrado)

contagem_uf_abs = df_filtrado["UF_nome"].value_counts().sort_values(ascending=False)
contagem_uf_rel = (contagem_uf_abs / total * 100).round(2)

contagem_regiao_abs = df_filtrado["Regiao"].value_counts().sort_values(ascending=False)
contagem_regiao_rel = (contagem_regiao_abs / total * 100).round(2)

# -----------------------------
# 5) Criar gráficos e salvar em PDF
# -----------------------------
with PdfPages("diagnostico_por_uf_regiao.pdf") as pdf:
    # Gráfico por UF
    plt.figure(figsize=(12,6))
    bars = contagem_uf_abs.plot(kind="bar")
    plt.title(f"Diagnóstico por Unidade da Federação (UF)\nTotal: {total}")
    plt.ylabel("Quantidade de casos (absoluta)")
    plt.xlabel("UF")
    plt.xticks(rotation=75, ha="right")

    # Colocar rótulos com valores absolutos e relativos
    for i, v in enumerate(contagem_uf_abs):
        plt.text(i, v + (v*0.01), f"{v} ({contagem_uf_rel.iloc[i]}%)", 
                 ha="center", va="bottom", fontsize=8, rotation=90)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Gráfico por Região
    plt.figure(figsize=(8,6))
    bars = contagem_regiao_abs.plot(kind="bar", color="orange")
    plt.title(f"Diagnóstico por Região\nTotal: {total}")
    plt.ylabel("Quantidade de casos (absoluta)")
    plt.xlabel("Região")
    plt.xticks(rotation=0)

    # Colocar rótulos com valores absolutos e relativos
    for i, v in enumerate(contagem_regiao_abs):
        plt.text(i, v + (v*0.01), f"{v} ({contagem_regiao_rel.iloc[i]}%)", 
                 ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

print("PDF gerado: diagnostico_por_uf_regiao.pdf")
