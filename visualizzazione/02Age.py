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

# Visualizzazione età senza contorno, con spazio tra barre e senza tick o linea X
plt.figure(figsize=(6, 4))
sns.histplot(
    df['Age'],
    bins=30,
    color="#12B22D",
    edgecolor=None,  # rimuove contorno
    shrink=0.9       # spazio tra barre
)
plt.title('Età', fontsize=12)
plt.xlabel('', fontsize=12)
plt.ylabel('', fontsize=12)

cx = plt.gca() 
cx.spines['top'].set_visible(False)
cx.spines['right'].set_visible(False)
cx.spines['left'].set_visible(False)
cx.spines['bottom'].set_visible(False)  # rimuove linea asse X
plt.tick_params(bottom=False, left=False)  # rimuove tick marks su X e Y

plt.show()
