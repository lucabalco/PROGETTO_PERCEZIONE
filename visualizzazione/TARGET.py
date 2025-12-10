import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo 

# 2. Unione di features e target
df = pd.read_csv("C:/Users/roma0/Desktop/UNI/Anno_3/Percezione/TentativoBinario/ObesityDataSet_raw_and_data_sinthetic.csv")
df.rename(columns={df.columns[-1]: 'Obesity_Level'}, inplace=True)

# 3. Creazione del BMI
df['BMI'] = df['Weight'] / (df['Height'] ** 2)


target_classes_1 = ['Normal_Weight', 'Insufficient_Weight']
df['Binary_Target'] = np.where(df['Obesity_Level'].isin(target_classes_1), 1, 0)

# Conteggi per plt.bar
counts = df['Binary_Target'].value_counts().sort_index()
indices = counts.index
heights = counts.values



plt.figure(figsize=(9, 5), facecolor="#ffe7c2")
plt.bar(indices, heights, width=0.4, color="#9D2007", align='center',edgecolor="#9D2007")
plt.xlabel('')
plt.xticks([0, 1], ['Sovrappeso/Obeso', 'Normopeso/Sottopeso'], 
        rotation=0, 
        ha='center', 
        fontsize=15,
        fontname="Prata", 
        color="#9C2007",
    ) 
plt.ylabel('') 

ax = plt.gca() 
ax.set_xlim(-1, 2)
ax.set_facecolor("#ffe7c2") 
ax.tick_params(axis='x', length=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.yaxis.grid(False) 
ax.set_yticks([])    


for i, h in zip(indices, heights):
    print(h)
    ax.text(i, h+50, f'{int(h)}', ha="center", fontsize=24, color="#FF914D", fontname="Prata", weight='bold' )
plt.savefig("C:/Users/roma0/Desktop/UNI/Anno_3/Percezione/target_isto/target.png")
plt.show()


plt.figure(figsize=(10, 10), facecolor="#ffe7c2")

plt.subplot(121)
sns.boxplot(df['Age'], color="#9D2007",notch=True,flierprops={"marker":"x"})
plt.title('Età', fontsize=24,fontname="Prata", color="#9C2007",weight= "bold")
plt.xlabel('')
plt.ylabel('Frequenza',labelpad=10, fontname="Prata", color="#9D2007", fontsize=12)
plt.yticks(color="#FF914D", fontname="Prata")
cx = plt.gca() 
cx.spines['top'].set_visible(False)
cx.spines['right'].set_visible(False)
cx.spines['left'].set_visible(False)
cx.spines['bottom'].set_visible(False)
cx.set_facecolor("#ffe7c2")
cx.tick_params(axis='x', length=0)


plt.subplot(122)
sns.boxplot(df['BMI'], color="#9D2007",notch=True,flierprops={"marker":"x"})
plt.title('BMI', fontsize=24,fontname="Prata", color="#9C2007",weight= "bold")
plt.xlabel("")
plt.ylabel('Frequenza',labelpad=10, fontname="Prata", color="#9D2007", fontsize=12)
plt.yticks(color="#FF914D", fontname="Prata")
cx= plt.gca()
cx.spines['top'].set_visible(False)
cx.spines['right'].set_visible(False)
cx.spines['left'].set_visible(False)
cx.spines['bottom'].set_visible(False)
cx.set_facecolor("#ffe7c2")
cx.tick_params(axis='x', length=0)


plt.savefig("C:/Users/roma0/Desktop/UNI/Anno_3/Percezione/target_isto/eta_BMI.png")
plt.show()
