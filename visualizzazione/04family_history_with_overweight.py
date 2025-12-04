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

# Visualizzazione countplot della feature 'family_history_with_overweight'
plt.figure(figsize=(6, 4))
ax = sns.countplot(
    data=df,
    x="family_history_with_overweight",
    hue="Peso_binario",
    hue_order=["Sovrappeso/Obeso", "Normopeso/Sottopeso"],
    palette=["#1E90FF", "#FF6B00"]
)

# Rimuovo spine superiori, laterali e inferiore
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)  # rimuove linea asse X

# Rimuovo le stanghette sui valori
ax.tick_params(axis='y', which='both', length=0)
ax.tick_params(axis='x', which='both', length=0)

# Mantengo solo i numeri sull'asse y
ax.tick_params(axis='y', labelleft=True)

# Etichetta asse y vuota
ax.set_ylabel("")

# Titolo centrato
plt.title("Famiglia con storia di obesità", fontsize=14, loc='center')
plt.xlabel("")

# Legenda fuori dal grafico
ax.legend(title="Categoria", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

