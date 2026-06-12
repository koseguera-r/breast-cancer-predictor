# ==================================================
# SELECCIÓN DE VARIABLES Y REGRESIÓN LOGÍSTICA
# Cáncer de mama - Wisconsin Diagnostic Breast Cancer
# ==================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
import statsmodels.api as sm
import pickle
import warnings
warnings.filterwarnings('ignore')

# ==================================================
# 1. CARGAR Y PREPARAR DATOS
# ==================================================
print("="*70)
print("1. CARGA DE DATOS")
print("="*70)
df = pd.read_csv('wdbc_dataset.csv')
df['target'] = (df['Diagnostico'] == 'M').astype(int)
feature_names = [col for col in df.columns if col not in ['ID', 'Diagnostico', 'target']]
X_raw = df[feature_names].copy()
y = df['target']
print(f"Dataset: {df.shape[0]} filas, {len(feature_names)} predictores")

# ==================================================
# 2. TRANSFORMACIÓN LOGARÍTMICA DE VARIABLES ASIMÉTRICAS
# ==================================================
print("\n2. TRANSFORMACIÓN LOG EN VARIABLES CON SKEW > 1.5")
skewness = X_raw.skew()
high_skew_vars = skewness[skewness > 1.5].index.tolist()
X_transformed = X_raw.copy()
for var in high_skew_vars:
    X_transformed[var] = np.log1p(X_transformed[var])
print(f"  Transformadas: {len(high_skew_vars)} variables")

# ==================================================
# 3. SELECCIÓN DE VARIABLES POR RFE (4 variables)
# ==================================================
print("\n3. SELECCIÓN DE VARIABLES (RFE con regresión logística)")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_transformed)
logreg = LogisticRegression(max_iter=1000, random_state=42)
rfe = RFE(estimator=logreg, n_features_to_select=4)
rfe.fit(X_scaled, y)
selected_vars = X_transformed.columns[rfe.support_]
print(f"  Variables seleccionadas: {list(selected_vars)}")

# ==================================================
# 4. ENTRENAMIENTO Y EVALUACIÓN DEL MODELO FINAL
# ==================================================
print("\n4. ENTRENAMIENTO Y EVALUACIÓN")
X_final = X_transformed[selected_vars]
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2,
                                                    random_state=42, stratify=y)
scaler_final = StandardScaler()
X_train_scaled = scaler_final.fit_transform(X_train)
X_test_scaled = scaler_final.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:,1]
acc = accuracy_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

print(f"  Accuracy: {acc:.4f}")
print(f"  AUC: {roc_auc:.4f}")
print(f"  Matriz de confusión:\n{confusion_matrix(y_test, y_pred)}")

# ==================================================
# 5. PSEUDO R² DE MCFADDEN
# ==================================================
print("\n5. PSEUDO R² DE MCFADDEN")
X_train_const = sm.add_constant(scaler_final.transform(X_train))
model_null = sm.Logit(y_train, np.ones(len(y_train))).fit(disp=0)
model_full = sm.Logit(y_train, X_train_const).fit(disp=0)
mcfadden = 1 - (model_full.llf / model_null.llf)
print(f"  McFadden R²: {mcfadden:.4f} ( >0.4 es excepcional)")

# ==================================================
# 6. GUARDAR MODELO Y ESCALADOR
# ==================================================
with open('modelo_final.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler_final.pkl', 'wb') as f:
    pickle.dump(scaler_final, f)
with open('variables_finales.txt', 'w') as f:
    f.write(str(list(selected_vars)))
print("\n Modelo guardado: modelo_final.pkl, scaler_final.pkl, variables_finales.txt")