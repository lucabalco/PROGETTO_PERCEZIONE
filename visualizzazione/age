import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Carica il dataset dal file CSV nella stessa cartella
print("Caricamento dataset in corso...")
df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

# Visualizza le prime righe
print("\nPrime righe del dataset:")
print(df.head())

# Arrotonda l'età per eccesso (round standard di Python)
df['Age_rounded'] = df['Age'].round()

# Visualizza statistiche sull'età
print("\n--- Statistiche Age ---")
print(f"Età minima: {df['Age_rounded'].min()}")
print(f"Età massima: {df['Age_rounded'].max()}")
print(f"Età media: {df['Age_rounded'].mean():.2f}")
print(f"Età mediana: {df['Age_rounded'].median()}")

# Crea l'istogramma
plt.figure(figsize=(12, 6))
plt.hist(df['Age_rounded'], bins=range(int(df['Age_rounded'].min()), int(df['Age_rounded'].max()) + 2), 
         color='#00C853', edgecolor='none', alpha=0.8, rwidth=0.8)

# Personalizza il grafico
plt.xlabel('Età (anni)', fontsize=12)
plt.ylabel('Frequenza', fontsize=12)

# Rimuovi le righe del contorno (spine)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.tight_layout()
plt.show()

# Mostra la distribuzione per fasce d'età
print("\n--- Distribuzione per fasce d'età ---")
bins_age = [0, 20, 30, 40, 50, 60, 100]
labels_age = ['<20', '20-29', '30-39', '40-49', '50-59', '60+']
df['Fascia_età'] = pd.cut(df['Age_rounded'], bins=bins_age, labels=labels_age, right=False)
print(df['Fascia_età'].value_counts().sort_index())