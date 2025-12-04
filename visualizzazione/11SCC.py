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

# Countplot per la feature SCC (Controllo delle calorie giornaliero)
plt.figure(figsize=(6, 4))
ax = sns.countplot(
    data=df,
    x="SCC",
    hue="Peso_binario",
    hue_order=["Sovrappeso/Obeso", "Normopeso/Sottopeso"],
    palette=["#1E90FF", "#FF6B00"]
)

# Rimuovo spine superiori, laterali e inferiore
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)  # rimuove linea asse X

# Rimuovo le stanghette dei tick sull'asse X
ax.tick_params(axis='x', which='both', bottom=False, top=False)

# Mantengo solo i numeri sull'asse y senza lineette
ax.yaxis.set_ticks_position('none')
ax.tick_params(axis='y', which='both', length=0)
ax.tick_params(axis='y', labelleft=True)

# Etichetta asse y vuota
ax.set_ylabel("")

# Titolo centrato
plt.title("Controllo delle calorie giornaliere", fontsize=14, loc='center')
plt.xlabel("")

# Legenda fuori dal grafico
ax.legend(title="Categoria", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

