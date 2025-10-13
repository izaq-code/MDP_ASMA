import pandas as pd
import matplotlib.pyplot as plt

# Carregar os dados
df = pd.read_csv("./Data/pns_2019.csv", sep=";", low_memory=False)

# Listas de variáveis
saudavel = [
    "Q00201","Q03001","Q055012","Q055013","Q055014","Q055016",
    "Q06307","Q06308","Q06309","Q06310","Q06311","Q068","Q075",
    "Q079","Q088","Q092","Q11007","Q11008","Q11009","Q11010",
    "Q11605","Q11606","Q11607","Q120","Q124"
]
asma = "Q074"
idade_col = "C008"

# Transformar as colunas em numéricas (ignora valores inválidos)
df[saudavel] = df[saudavel].apply(pd.to_numeric, errors='coerce')
df[asma] = pd.to_numeric(df[asma], errors='coerce')
df[idade_col] = pd.to_numeric(df[idade_col], errors='coerce')

# Filtrar idade entre 15 e 65 anos
df = df[(df[idade_col] >= 15) & (df[idade_col] <= 65)]

# Criar coluna Saudavel: True se respondeu 2 (Não) em pelo menos uma pergunta saudável
df['Saudavel'] = df[saudavel].eq(2).any(axis=1)

# Criar coluna DoenteAsma: True se respondeu 1 (Sim) no Q074
df['DoenteAsma'] = df[asma] == 1

# Contagem das categorias
contagem = {
    "Saudáveis": df['Saudavel'].sum(),
    "Doentes (Asma)": df['DoenteAsma'].sum()
}

# Criar gráfico de barras
plt.figure(figsize=(8,6))
plt.bar(contagem.keys(), contagem.values(), color=['green','red'], edgecolor='black')
plt.title("Comparação Saudáveis vs Doentes (Asma) - Idade 15-65 - PNS 2019")
plt.ylabel("Quantidade de pessoas")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Salvar como PNG
plt.savefig("saudaveis_vs_asma_15_65.png", dpi=300, bbox_inches='tight')

# Mostrar gráfico
plt.show()
