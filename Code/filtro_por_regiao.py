import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1) Leitura dos dados
# -----------------------------
arquivo = "./Data/pns_2019.csv"  
df = pd.read_csv(arquivo, sep=";", low_memory=False)

# -----------------------------
# 2) População total e filtrada
# -----------------------------
total_populacao = len(df)              
df_filtrado = df[df["Q074"] == 1]      
total_diagnosticados = len(df_filtrado)

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

df_filtrado["UF_nome"] = df_filtrado["V0001"].map(lambda x: ufs.get(x, ("Desconhecido", "Outro"))[0])
df_filtrado["Regiao"] = df_filtrado["V0001"].map(lambda x: ufs.get(x, ("Desconhecido", "Outro"))[1])

# -----------------------------
# 4) Contagens
# -----------------------------
contagem_uf_abs = df_filtrado["UF_nome"].value_counts().sort_values(ascending=False)
contagem_uf_rel_pop = (contagem_uf_abs / total_populacao * 100).round(2)

contagem_regiao_abs = df_filtrado["Regiao"].value_counts().sort_values(ascending=False)
contagem_regiao_rel_pop = (contagem_regiao_abs / total_populacao * 100).round(2)

# -----------------------------
# 5) Criar gráficos e salvar como PNG
# -----------------------------

# ABSOLUTO POR UF
plt.figure(figsize=(12,6))
contagem_uf_abs.plot(kind="bar")
plt.title(f"Diagnóstico por UF (Absoluto)\nTotal diagnosticados: {total_diagnosticados}")
plt.ylabel("Quantidade de casos")
plt.xlabel("UF")
plt.xticks(rotation=75, ha="right")
for i, v in enumerate(contagem_uf_abs):
    plt.text(i, v + (v*0.01), str(v), ha="center", va="bottom", fontsize=8, rotation=90)
plt.tight_layout()
plt.savefig("uf_absoluto.png", dpi=300)
plt.close()

# RELATIVO POR UF
plt.figure(figsize=(12,6))
contagem_uf_rel_pop.plot(kind="bar", color="green")
plt.title(f"Diagnóstico por UF (Relativo à População Total)\nPopulação total: {total_populacao}")
plt.ylabel("Percentual da população (%)")
plt.xlabel("UF")
plt.xticks(rotation=75, ha="right")
for i, v in enumerate(contagem_uf_rel_pop):
    plt.text(i, v + 0.01, f"{v}%", ha="center", va="bottom", fontsize=8, rotation=90)
plt.tight_layout()
plt.savefig("uf_relativo.png", dpi=300)
plt.close()

# ABSOLUTO POR REGIÃO
plt.figure(figsize=(8,6))
contagem_regiao_abs.plot(kind="bar", color="orange")
plt.title(f"Diagnóstico por Região (Absoluto)\nTotal diagnosticados: {total_diagnosticados}")
plt.ylabel("Quantidade de casos")
plt.xlabel("Região")
plt.xticks(rotation=0)
for i, v in enumerate(contagem_regiao_abs):
    plt.text(i, v + (v*0.01), str(v), ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig("regiao_absoluto.png", dpi=300)
plt.close()

# RELATIVO POR REGIÃO
plt.figure(figsize=(8,6))
contagem_regiao_rel_pop.plot(kind="bar", color="red")
plt.title(f"Diagnóstico por Região (Relativo à População Total)\nPopulação total: {total_populacao}")
plt.ylabel("Percentual da população (%)")
plt.xlabel("Região")
plt.xticks(rotation=0)
for i, v in enumerate(contagem_regiao_rel_pop):
    plt.text(i, v + 0.01, f"{v}%", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig("regiao_relativo.png", dpi=300)
plt.close()

print("Gráficos salvos como PNG individualmente!")
