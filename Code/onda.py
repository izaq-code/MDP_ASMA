import pandas as pd
import matplotlib.pyplot as plt

# Carregar os dados (troque pelo caminho real da PNS 2019)
df = pd.read_csv("./Data/pns_2019.csv", sep=";")  

# Variáveis
saudavel = [
    "Q00201","Q03001","Q055012","Q055013","Q055014","Q055016",
    "Q06307","Q06308","Q06309","Q06310","Q06311","Q068","Q075",
    "Q079","Q088","Q092","Q11007","Q11008","Q11009","Q11010",
    "Q11605","Q11606","Q11607","Q120","Q124"
]
asma = "Q074"

# Definir saudável: respondeu 2 em todas as perguntas "saudável"
df["Saudavel"] = df[saudavel].eq(2).all(axis=1)

# Definir doente (asma): respondeu 1 no Q074
df["Asma"] = df[asma].eq(1)

# Contagem
contagem = {
    "Saudável": df["Saudavel"].sum(),
    "Asma": df["Asma"].sum()
}

# Criar gráfico de barras
plt.bar(contagem.keys(), contagem.values())
plt.title("Distribuição de Saúde (PNS 2019)")
plt.ylabel("Número de pessoas")
plt.show()
