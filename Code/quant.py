import pandas as pd

# Carregar os dados (substitua pelo nome real do arquivo CSV da PNS)
df = pd.read_csv("./Data/pns_2019.csv", sep=";")  

# Lista das variáveis de câncer
variaveis_cancer = [

    "Q11008",   # Câncer de pele
    "Q088",   # Câncer de pele tipo melanoma
    "Q079",   # Câncer de pele
    "Q074",   # Câncer de pele tipo melanoma
]

# Contar a quantidade de respostas positivas (geralmente codificadas como 1 = Sim)
resultados = {}
for var in variaveis_cancer:
    if var in df.columns:
        resultados[var] = df[var].value_counts().get(1, 0)  # conta os "Sim"
    else:
        resultados[var] = "Variável não encontrada"

# Transformar em DataFrame para visualização
df_resultados = pd.DataFrame.from_dict(resultados, orient="index", columns=["Qtd Diagnóstico"])
print(df_resultados)

# Salvar em Excel
df_resultados.to_excel("quantidade_diagnosticos_cancer.xlsx")
