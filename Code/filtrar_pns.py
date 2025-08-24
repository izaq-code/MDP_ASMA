import pandas as pd

# Caminho do arquivo (ajuste a extensão conforme seu caso: .csv ou .xlsx)
arquivo = "./Data/pns_2019.csv"   # ou "pns_2019.xlsx"

# --- Leitura do arquivo ---
# Se for CSV separado por ";"
df = pd.read_csv(arquivo, sep=";", low_memory=False)

# Se for Excel, comente a linha acima e use essa:
# df = pd.read_excel(arquivo)

# --- Filtrar somente quem marcou 1 na coluna Q06307 ---
df_filtrado = df[df["Q068"] == 1]

# --- Exportar para novo arquivo ---
df_filtrado.to_csv("pns_2019_filtrado-avc.csv", sep=";", index=False)

print("Filtro concluído! Arquivo salvo como 'pns_2019_filtrado-avc.csv'")
