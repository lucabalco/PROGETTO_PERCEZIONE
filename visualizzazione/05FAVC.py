import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Caricamento dataset
df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")

# Creazione colonna binaria Normopeso / Non normopeso
normopeso_classi = ["Normal_Weight", "Insufficient_Weight"]
df["Peso_binario"] = df["NObeyesdad"].apply(
    lambda x: "Normopeso/Sottopeso" if x in normopeso_classi else "Sovrappeso/Obeso"
)

# Visualizzazione countplot della feature FAVC
plt.figure(figsize=(6, 4))

ax = sns.countplot(
    data=df,
    x="FAVC",
    hue="Peso_binario",
    hue_order=["Sovrappeso/Obeso", "Normopeso/Sottopeso"],
    palette=["#1E90FF", "#FF6B00"]
)

# Rimuovo spine superiori, laterali e inferiore
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)  # rimuove linea asse X

# Rimuovo stanghette sui tick
ax.tick_params(bottom=False, left=False)

# Etichetta asse y vuota
ax.set_ylabel("")

# Titolo centrato
plt.title("Consumazione di cibo altamente calorici", fontsize=14, loc='center')
plt.xlabel("")

# Legenda fuori dal grafico
ax.legend(title="Categoria", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()
