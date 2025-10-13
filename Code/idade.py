import pandas as pd
import matplotlib.pyplot as plt

# Carregar dados
df = pd.read_csv("./Data/pns_2019.csv", sep=";", low_memory=False)

# Converter colunas para numérico
df["Q074"] = pd.to_numeric(df["Q074"], errors='coerce')  # Asma
df["Q075"] = pd.to_numeric(df["Q075"], errors='coerce')  # Idade

# Filtrar pessoas com asma e remover idades faltantes
df_asma = df[df["Q074"] == 1].dropna(subset=["Q075"])

# Calcular frequência absoluta simples
freq_simples = df_asma["Q075"].value_counts().sort_index()

# Calcular frequência acumulada simples
freq_acumulada = freq_simples.cumsum()

# Criar tabela organizada
tabela_freq = pd.DataFrame({
    "Idade": freq_simples.index,
    "Frequência Absoluta": freq_simples.values,
    "Frequência Acumulada": freq_acumulada.values
})

print(tabela_freq.head(15))  # mostra as 15 primeiras idades

# Plotar gráfico de linha da frequência acumulada
plt.figure(figsize=(10,6))
plt.plot(freq_acumulada.index, freq_acumulada.values, color="blue", linewidth=2, marker="o")

# Personalização do gráfico
plt.title("Frequência Acumulada Simples de Pessoas com Asma por Idade - PNS 2019")
plt.xlabel("Idade (anos)")
plt.ylabel("Frequência acumulada (número de pessoas)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Salvar e mostrar
plt.savefig("frequencia_acumulada_simples_asma.png", dpi=300)
plt.show()
