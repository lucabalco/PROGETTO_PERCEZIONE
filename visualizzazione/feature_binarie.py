import pandas as pd
import matplotlib.pyplot as plt

# 1. Caricamento dataset
df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")

# 2. Creazione colonna binaria Normopeso / Non normopeso
normopeso_classi = ["Normal_Weight", "Insufficient_Weight"]
df["Peso_binario"] = df["NObeyesdad"].apply(
    lambda x: "Normopeso/Sottopeso" if x in normopeso_classi else "Sovrappeso/Obeso"
)

# 3. Seleziona solo le feature binarie
binary_features = [col for col in df.columns if df[col].nunique() == 2 and col != "Peso_binario"]

titoli = ['Genere', 'Famiglia con storia di obesità', 'Consumazione di cibo altamente calorici', 'Fumatore', 'Controllo delle calorie giornaliero']

# 4. Grafici manuali con valori sopra le barre
for i, feature in enumerate(binary_features):
    counts = pd.crosstab(df[feature], df["Peso_binario"])
    
    x = range(len(counts))
    width = 0.4
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    bars1 = ax.bar([p - width/2 for p in x], counts["Sovrappeso/Obeso"], width=width, color="#1E90FF", label="Sovrappeso/Obeso")
    bars2 = ax.bar([p + width/2 for p in x], counts["Normopeso/Sottopeso"], width=width, color="#FF6B00", label="Normopeso/Sottopeso")
    
    # Aggiungi valori sopra le barre
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.3,
            int(height),
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    # Etichette x
    ax.set_xticks(x)
    ax.set_xticklabels(counts.index)
    
    # Rimuovo spine e asse y
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.yaxis.set_visible(False)
    
    # Rimuovo ticks asse x e minor ticks
    ax.tick_params(axis='x', which='both', length=0)
    ax.minorticks_off()
    
    # Limite asse y per sicurezza
    ax.set_ylim(0, max(counts.max())*1.15)
    
    # Titolo centrato
    plt.title(titoli[i], fontsize=14, loc='center')
    
    # Legenda fuori dal grafico
    ax.legend(title="Categoria", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()
