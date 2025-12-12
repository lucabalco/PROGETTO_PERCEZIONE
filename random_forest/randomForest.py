import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import seaborn as sns

# --- Modifica: Caricamento dataset da file CSV locale ---
print("Caricamento dataset da file CSV locale...")

# Assumendo che il file sia nella stessa directory dello script o accessibile
df = pd.read_csv("C:/Users/roma0/Desktop/UNI/Anno_3/Percezione/TentativoBinario/ObesityDataSet_raw_and_data_sinthetic.csv")
print("Dataset caricato con successo.")

# Separazione Features (X) e Target (y)
X = df.drop(columns=['NObeyesdad'])
y = df[['NObeyesdad']]

print("\nShape originale:")
print(f"Features: {X.shape}")
print(f"Target: {y.shape}")

# Rimozione delle feature peso, altezza ed età
print("\nRimozione di Weight, Height e Age...")
features_to_drop = ['Weight', 'Height', 'Age']
# Usiamo df.columns per verificare le colonne esistenti prima del drop (caso in cui Age non sia presente)
columns_to_drop = [col for col in features_to_drop if col in X.columns]
X = X.drop(columns=columns_to_drop)

print(f"Nuove features: {X.columns.tolist()}")
print(f"Shape dopo rimozione: {X.shape}")

# Conversione target binario - INVERSA RISPETTO ALLA RICHIESTA PRECEDENTE
# 1 = sovrappeso/obeso (Overweight_Level_I, Overweight_Level_II, Obesity_Type_I, Obesity_Type_II, Obesity_Type_III)
# 0 = sottopeso/normopeso (Insufficient_Weight, Normal_Weight)
print("\nConversione target binario (inversa)...")
print(f"Classi originali: {y['NObeyesdad'].unique()}")

def convert_target(label):
    if label in ['Insufficient_Weight', 'Normal_Weight']:
        return 0  # sottopeso/normopeso
    else:
        return 1  # sovrappeso/obeso

y_binary = y['NObeyesdad'].apply(convert_target)

print(f"\nDistribuzione classi (Target Invertito):")
print(f"Classe 1 (Sovrappeso/Obeso): {sum(y_binary == 1)}")
print(f"Classe 0 (Sottopeso/Normopeso): {sum(y_binary == 0)}")

# Encoding delle variabili categoriche
print("\nEncoding variabili categoriche...")
le_dict = {}
X_encoded = X.copy()

for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col])
        le_dict[col] = le
        print(f"{col}: {list(le.classes_)}")

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_binary, test_size=0.2, random_state=3, stratify=y_binary
)

print(f"\nTrain set: {X_train.shape[0]} campioni")
print(f"Test set: {X_test.shape[0]} campioni")

# Creazione e addestramento Random Forest
print("\n" + "="*60)
print("TRAINING RANDOM FOREST (TARGET INVERTITO)")
print("="*60)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=1,
    random_state=3,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
print("Modello addestrato con successo")

# Predizioni
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1] # Probabilità della CLASSE 1 (Sovrappeso/Obeso)

# Predizioni anche sul training set per confronto ROC
y_train_pred_proba = rf_model.predict_proba(X_train)[:, 1]

# Metriche
print("\n" + "="*60)
print("RISULTATI")
print("="*60)

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")

print("\nClassification Report:")
# Nomi del target invertiti: 0 è Sottopeso/Normopeso, 1 è Sovrappeso/Obeso
target_names = ['Sottopeso/Normopeso', 'Sovrappeso/Obeso'] 
print(classification_report(y_test, y_pred, target_names=target_names))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Cross-validation
print("\nCross-validation (5-fold):")
cv_scores = cross_val_score(rf_model, X_encoded, y_binary, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Feature importance
print("\n" + "="*60)
print("FEATURE IMPORTANCE")
print("="*60)

feature_importance = pd.DataFrame({
    'feature': X_encoded.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 feature più importanti:")
print(feature_importance.head(10).to_string(index=False))

# --- Visualizzazione risultati - Grafici separati ---

colors = ["#ffd1a6", '#b65337' ]
gradiente = LinearSegmentedColormap.from_list(
    "Mappa_Personalizzata",
    colors,
    N=256 
)


# Grafico 1: Confusion Matrix
# 1. Creazione della figura con un asse per la heatmap (ax)
fig = plt.figure(figsize=(7, 7), facecolor= '#fff7eb')
ax = fig.add_subplot(111)
cbar_ax = fig.add_axes([0.90, 0.15, 0.03, 0.7]) 
sns.heatmap(
    cm, 
    ax=ax, 
    cbar_ax=cbar_ax, 
    annot=True, 
    fmt='d', 
    cmap=gradiente,
    annot_kws = {'fontname':"Prata", 'color':"#373737", 'fontsize':'16', 'weight':'bold'}
)

cb = fig.colorbar(ax.collections[0], cax=cbar_ax)
cb.outline.set_edgecolor('#fff7eb')
for label in cb.ax.yaxis.get_ticklabels():
    label.set_color('#b65337')
    label.set_fontname("Prata") 
    label.set_fontsize(12)      
    label.set_fontweight('bold')

ax.set_ylabel('Vero', fontname="Prata", color="#b65337", fontsize=14, weight='bold')
ax.set_xlabel('Predetto', fontname="Prata", color="#b65337", fontsize=14, weight='bold')

ax.tick_params(axis='y', length=0)
ax.get_yaxis().set_ticks([]) 
ax.tick_params(axis='x', length=0)
ax.get_xaxis().set_ticks([]) 

ax.set_xticks([0.5, 1.5], target_names, fontname="Prata", color="#b65337", fontsize=14, weight='bold')
ax.set_yticks([0.5, 1.5], target_names, fontname="Prata", color="#b65337", fontsize=14, weight='bold')

plt.tight_layout(rect=[0, 0, 0.9, 1]) 
plt.savefig("C:/Users/roma0/Desktop/UNI/Anno_3/Percezione/img_randomForest/Confusion Matrix.png")
plt.show()

# Grafico ROC: Curva ROC per Training e Test
print("\nCalcolo curva ROC per training e test...")


fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_pred_proba)
roc_auc_test = auc(fpr_test, tpr_test)
fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_pred_proba)
roc_auc_train = auc(fpr_train, tpr_train)

plt.figure(figsize=(10, 8), facecolor= '#fff7eb')

# Curva ROC Training
plt.plot(fpr_train, tpr_train, color='blue', lw=2, 
          label=f'Training ROC (AUC = {roc_auc_train:.4f})')

# Curva ROC Test
plt.plot(fpr_test, tpr_test, color='green', lw=2, 
          label=f'Test ROC (AUC = {roc_auc_test:.4f})')

# Linea diagonale (classificatore casuale)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', 
          label='Classificatore Casuale (AUC = 0.5)')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

plt.tick_params(axis='x', labelcolor='#b65337', labelsize=12, size=0, pad=10, labelfontfamily='Prata')
plt.tick_params(axis='y', labelcolor='#b65337', labelsize=12, size=0, pad=10,  labelfontfamily='Prata')

plt.xlabel('False Positive Rate (FPR)', fontname="Prata", color="#b65337", fontsize=14, weight='bold')
plt.ylabel('True Positive Rate (TPR)', fontname="Prata", color="#b65337", fontsize=14, weight='bold')
plt.title('Curva ROC - Training vs Test', fontname="Prata", color="#b65337", fontsize=18, weight='bold')

# Aggiungi punto ottimale sul test set
optimal_idx = np.argmax(tpr_test - fpr_test)
optimal_threshold = thresholds_test[optimal_idx]
plt.plot(fpr_test[optimal_idx], tpr_test[optimal_idx], 'ro', markersize=10, 
          label=f'Punto Ottimale (soglia={optimal_threshold:.3f})')

legend_font = FontProperties(family='Prata', size=11, weight='bold')
legend = plt.legend(loc="lower right", prop=legend_font, facecolor='#fff7eb')
for text in legend.get_texts():
    text.set_color('#b65337')
    
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.savefig("C:/Users/roma0/Desktop/UNI/Anno_3/Percezione/img_randomForest/ROC.png")
plt.show()

print(f"\nROC AUC Score Training: {roc_auc_train:.4f}")
print(f"ROC AUC Score Test: {roc_auc_test:.4f}")
print(f"Differenza (overfitting check): {abs(roc_auc_train - roc_auc_test):.4f}")

if abs(roc_auc_train - roc_auc_test) < 0.05:
    print("OK - Modello ben bilanciato (poco overfitting)")
elif abs(roc_auc_train - roc_auc_test) < 0.10:
    print("ATTENZIONE - Leggero overfitting")
else:
    print("ATTENZIONE - Overfitting significativo")

# Grafico 5: Visualizzazione di un albero decisionale
print("\nVisualizzazione di un albero decisionale dal Random Forest...")
plt.figure(figsize=(14, 8), facecolor = '#fff7eb')
tree_to_plot = rf_model.estimators_[23]

albero = plot_tree(tree_to_plot, 
                   feature_names=X_encoded.columns,
                   class_names=target_names, 
                   rounded=False,
                   fontsize=6,
                   max_depth=3)
for text in albero:
    text.set_color('#000000')
    text.set_fontweight('bold')
    text.set_alpha(1.0)
    bbox = text.get_bbox_patch() 
    if bbox:
        bbox.set_edgecolor('#000000')
        bbox.set_facecolor('#ffbc7e') 
        bbox.set_linewidth(1) 

plt.tight_layout()
plt.savefig("C:/Users/roma0/Desktop/UNI/Anno_3/Percezione/img_randomForest/Albero_Completo.png")
plt.show()

# Grafico 2: Feature Importance
plt.figure(figsize=(10, 6), fontcolor='#fff7eb')
top_features = feature_importance.head(10)
plt.barh(top_features['feature'], top_features['importance'], color='#b65337')
plt.xlabel('Importanza', fontsize=12)
plt.title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig("C:/Users/roma0/Desktop/UNI/Anno_3/Percezione/img_randomForest/Feature_Importance.png")
plt.show()

# Grafico 3: Distribuzione predizioni
plt.figure(figsize=(10, 6))
# Il primo istogramma corrisponde a y_test == 0 (Sottopeso/Normopeso)
# Il secondo istogramma corrisponde a y_test == 1 (Sovrappeso/Obeso)
plt.hist([y_pred_proba[y_test == 0], y_pred_proba[y_test == 1]], 
          bins=30, label=['Sottopeso/Normopeso', 'Sovrappeso/Obeso'], 
          alpha=0.7, color=['#1f77b4', '#ff7f0e']) # Scambiati i colori per coerenza
plt.xlabel('Probabilità predetta (classe 1 - Sovrappeso/Obeso)', fontsize=12)
plt.ylabel('Frequenza', fontsize=12)
plt.title('Distribuzione Probabilità Predette', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("C:/Users/roma0/Desktop/UNI/Anno_3/Percezione/img_randomForest/Distrbuzione predizioni.png")
plt.show()

# Grafico 4: Cross-validation scores
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cv_scores) + 1), cv_scores, marker='o', 
          linewidth=2, markersize=8, color='steelblue')
plt.axhline(y=cv_scores.mean(), color='r', linestyle='--', 
            linewidth=2, label=f'Media: {cv_scores.mean():.4f}')
plt.xlabel('Fold', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Cross-Validation Scores', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim([cv_scores.min() - 0.01, cv_scores.max() + 0.01])
plt.tight_layout()
plt.savefig("C:/Users/roma0/Desktop/UNI/Anno_3/Percezione/img_randomForest/Cross Validation.png")
plt.show()
