import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo

# Fetch Dataset
estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition = fetch_ucirepo(id=544)
X = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.features

colonne_richieste = ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
dati_selezionati = X[colonne_richieste].copy() 

# Arrotondamento e Limitazione
dati_selezionati = np.ceil(dati_selezionati - 0.5).astype(int)
dati_selezionati = dati_selezionati.clip(upper=4)
sns.set_style("white") 

# Dizionario che contiene i dettagli per ogni grafico
plot_details = {
    'FCVC': {
        'title': 'Frequenza Consumo di Vegetali', 
        'labels': {1: 'Mai', 2: 'A volte', 3: 'Sempre'}
    },
    'NCP': {
        'title': 'Numero di Pasti Principali', 
        'labels': {1: '1', 2: '2', 3: '3', 4: 'Più di 3'} 
    },
    'CH2O': {
        'title': 'Consumo d\'Acqua', 
        'labels': {1: 'Meno di 1L', 2: '1-2L', 3: 'Più di 2L'} 
    },
    'FAF': {
        'title': 'Frequenza Attività Fisica', 
        'labels': {0: 'Mai', 1: '1-2 volte/sett.', 2: '2-4 volte/sett.', 3: '4-5 volte/sett.'}
    },
    'TUE': {
        'title': 'Uso di Dispositivi Tecnologici', 
        'labels': {0: '0-2 ore', 1: '3-5 ore', 2: 'Più di 5 ore'}
    }
}

# Generazione degli istogrammi  
for i, (col, details) in enumerate(plot_details.items()):    
    plt.figure(figsize=(5, 5))
    sns.histplot(
        data=dati_selezionati,
        x=col,
        bins=np.arange(0.5, 5.5, 1),
        discrete=True,
        color="#12B22D" , 
        shrink=0.5 
    )
    #Styling
    plt.title(f"{details['title']}", fontsize=14)
    plt.ylabel('')
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    tick_positions = list(details['labels'].keys())
    tick_labels = list(details['labels'].values())
    
    plt.xticks(tick_positions, tick_labels, ha='center', fontsize=10)
    plt.xlabel('')
    plt.tight_layout()

    plt.show()
