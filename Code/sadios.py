import pandas as pd
import matplotlib.pyplot as plt

# Carregar os dados
df = pd.read_csv("./Data/pns_2019.csv", sep=";", low_memory=False)

# Variáveis
saudavel = [
    "Q00201","Q03001","Q055012","Q055013","Q055014","Q055016",
    "Q06307","Q06308","Q06309","Q06310","Q06311","Q068","Q075",
    "Q079","Q088","Q092","Q11007","Q11008","Q11009","Q11010",
    "Q11605","Q11606","Q11607","Q120","Q124"
]
asma = "Q074"
idade_col = "C008"

# Converter valores para numérico
df[saudavel] = df[saudavel].apply(pd.to_numeric, errors='coerce')
df[asma] = pd.to_numeric(df[asma], errors='coerce')
df[idade_col] = pd.to_numeric(df[idade_col], errors='coerce')

# Filtrar idade entre 0 e 30 anos
df = df[(df[idade_col] >= 0) & (df[idade_col] <= 30)]

# Lógica de saúde: Saudável se respondeu 2 (não) em ao menos uma pergunta
df['Saudavel'] = df[saudavel].eq(2).any(axis=1)

# Lógica de doença (asma): respondeu 1 (sim)
df['DoenteAsma'] = df[asma] == 1

# Contagens
contagem = {
    "Saudáveis": df['Saudavel'].sum(),
    "Doentes (Asma)": df['DoenteAsma'].sum()
}

# ---- PRINT NO CMD ----
print("\n=== RESULTADO (Idade 0–30) ===")
for categoria, valor in contagem.items():
    print(f"{categoria}: {valor}")
print("==============================\n")

# Gráfico
plt.figure(figsize=(8,6))
plt.bar(contagem.keys(), contagem.values(), color=['green','red'], edgecolor='black')
plt.title("Comparação Saudáveis vs Doentes (Asma) - Idade 0-30 - PNS 2019")
plt.ylabel("Quantidade de pessoas")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig("saudaveis_vs_asma_0_30.png", dpi=300, bbox_inches='tight')
plt.show()
