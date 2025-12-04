import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo 

# 1. Caricamento del dataset
estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition = fetch_ucirepo(id=544) 
X = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.features 
y = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.targets 

# 2. Unione di features e target
df = pd.concat([X, y], axis=1)
df.rename(columns={df.columns[-1]: 'Obesity_Level'}, inplace=True)

# 3. Creazione del BMI
df['BMI'] = df['Weight'] / (df['Height'] ** 2)

# 4. Binarizzazione della variabile target
target_classes_1 = ['Normal_Weight', 'Insufficient_Weight']
df['Binary_Target'] = np.where(df['Obesity_Level'].isin(target_classes_1), 1, 0)

# Conteggi per plt.bar
counts = df['Binary_Target'].value_counts().sort_index()
indices = counts.index
heights = counts.values

# --- Inizializzazione per la Visualizzazione ---
sns.set_style("white") 
plt.rcParams['figure.figsize'] = (7, 5)

## Visualizzazione dei Dati

### 1. Distribuzione della Nuova Classe Target Binaria (Stretta) 📊

plt.figure(figsize=(6, 4))

# Usiamo una larghezza stretta (width=0.4)
BAR_WIDTH = 0.4
plt.bar(indices, heights, width=BAR_WIDTH, color=["#1E90FF", "#FF6B00"], align='center')

ax = plt.gca() 


# **********************************************
# MODIFICHE FINALI: Barre Strette + Visualizzazione Stretta
# **********************************************

# Stringere i limiti dell'asse X per stringere la visualizzazione complessiva
ax.set_xlim(-1, 2) 

# Mantenimento delle modifiche precedenti:
plt.xlabel('') 
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.xticks([0, 1], ['Sovrappeso/Obeso', 'Normopeso/Sottopeso'])
ax.yaxis.grid(False) 
ax.set_yticks([])    
plt.ylabel('') 

# Aggiunta del conteggio sopra le barre
for i, h in zip(indices, heights):
    ax.text(i, h + 10, f'{int(h)}', ha="center", fontsize=11, weight='bold')

plt.show()
featureC = ['Age','BMI']
featureC_descrizione = ['Età','BMI']
for i in range(len(featureC)) : 
    plt.figure(figsize=(6, 4))
    sns.histplot(df[featureC[i]], bins=30, color="#12B22D")
    plt.title(featureC_descrizione[i], fontsize=12)
    plt.xlabel('Valore', fontsize=12)
    plt.ylabel('', fontsize=12)
    cx = plt.gca() 
    cx.spines['top'].set_visible(False)
    cx.spines['right'].set_visible(False)
    cx.spines['left'].set_visible(False)
    plt.show()
