import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo 

# Caricamento del dataset
estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition = fetch_ucirepo(id=544) 
X = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.features 
y = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.targets 

# Unione di features e target
df = pd.concat([X, y], axis=1)
df.rename(columns={df.columns[-1]: 'Obesity_Level'}, inplace=True)

# Creazione del BMI
df['BMI'] = df['Weight'] / (df['Height'] ** 2)

# Binarizzazione della variabile target
target_classes_1 = ['Normal_Weight', 'Insufficient_Weight']
df['Binary_Target'] = np.where(df['Obesity_Level'].isin(target_classes_1), 1, 0)

# Conteggi per plt.bar
counts = df['Binary_Target'].value_counts().sort_index()
indices = counts.index
heights = counts.values

# Visualizzazione del target
sns.set_style("white") 
plt.rcParams['figure.figsize'] = (7, 5)
plt.figure(figsize=(6, 4))
BAR_WIDTH = 0.4
plt.bar(indices, heights, width=BAR_WIDTH, color=["#1E90FF", "#FF6B00"], align='center')
ax = plt.gca() 
ax.set_xlim(-1, 2) 
plt.xlabel('') 
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)   # rimuove linea asse X
plt.xticks([0, 1], ['Sovrappeso/Obeso', 'Normopeso/Sottopeso'])
ax.yaxis.grid(False) 
ax.set_yticks([])    
plt.ylabel('') 
for i, h in zip(indices, heights):
    ax.text(i, h + 10, f'{int(h)}', ha="center", fontsize=11, weight='bold')
plt.show()
