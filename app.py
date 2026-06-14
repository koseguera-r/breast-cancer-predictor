"""
OncoPredict - Dashboard clínico para predicción de cáncer de mama
Modelo de regresión logística con visualización interactiva estilo publicación médica.
Autor: Kevin Yael Oseguera Reyes
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# ==================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==================================================
st.set_page_config(
    page_title="OncoPredict · Cáncer de Mama",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)
# Imagen de cabecera (lazo rosa)
col_logo, col_title = st.columns([1, 5])
with col_logo:
        st.image("assets/pink_ribbon.png", width=80)  
with col_title:
    st.title("🔬 OncoPredict · Apoyo al Diagnóstico de Cáncer de Mama")

# Ocultar la barra de herramientas de Plotly (botones de zoom, etc.)
config_plotly = {'displayModeBar': False}
st.markdown("""
<style>
    /* Botón rosa (no afecta texto) */
    .stButton > button {
        background-color: #E83E8C !important;
        color: white !important;
        border-radius: 40px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        border: none !important;
    }
    .stButton > button:hover {
        background-color: #C2185B !important;
    }
    /* Tarjetas de diagnóstico (solo fondo y borde, no texto) */
    .diagnostico-benigno {
        background-color: #E6F7F0;
        border-left: 8px solid #2E86AB;
        padding: 1rem;
        border-radius: 16px;
    }
    .diagnostico-maligno {
        background-color: #FDECEA;
        border-left: 8px solid #D9534F;
        padding: 1rem;
        border-radius: 16px;
    }
    .disclaimer {
        font-size: 0.7rem;
        text-align: center;
        margin-top: 2rem;
        border-top: 1px solid #ccc;
        padding-top: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ==================================================
# CARGA DE MODELO, ESCALADOR Y VARIABLES
# ==================================================
@st.cache_resource
def load_artifacts():
    with open('modelo_final.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler_final.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('variables_finales.txt', 'r') as f:
        vars_list = eval(f.read())
    return model, scaler, vars_list

model, scaler, FEATURES = load_artifacts()

# Cargar dataset original
@st.cache_data
def load_data():
    df = pd.read_csv('wdbc_dataset.csv')
    df['target'] = (df['Diagnostico'] == 'M').astype(int)
    return df

df = load_data()
# ==================================================
# INICIALIZACIÓN DEL ESTADO DE LA SESIÓN
# ==================================================
if 'predecir' not in st.session_state:
    st.session_state.predecir = False
if 'area2' not in st.session_state:
    st.session_state.area2 = df['area2'].median()
if 'radius3' not in st.session_state:
    st.session_state.radius3 = df['radius3'].median()
if 'area3' not in st.session_state:
    st.session_state.area3 = df['area3'].median()
if 'concave_points3' not in st.session_state:
    st.session_state.concave_points3 = df['concave_points3'].median()

# Aplicar transformación logarítmica a las variables asimétricas (como en el entrenamiento)
@st.cache_data
def preprocess_data(df):
    X_raw = df[FEATURES].copy()
    # Identificar skew > 1.5 en el dataset original para aplicar log1p
    skewness = X_raw.skew()
    high_skew_vars = skewness[skewness > 1.5].index.tolist()
    X_transformed = X_raw.copy()
    for var in high_skew_vars:
        X_transformed[var] = np.log1p(X_transformed[var])
    return X_transformed, df['target']

X_transformed, y = preprocess_data(df)

# Dividir en train/test para métricas (misma semilla que entrenamiento)
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42, stratify=y)
# Escalar con el scaler guardado
X_test_scaled = scaler.transform(X_test)

# Predicciones y métricas
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]
acc = accuracy_score(y_test, y_pred)
sens = recall_score(y_test, y_pred)
spec = recall_score(y_test, y_pred, pos_label=0)
prec = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Coeficientes del modelo con p-values usando statsmodels (sobre datos escalados)
X_train_scaled = scaler.transform(X_train)
X_train_const = sm.add_constant(X_train_scaled)
logit_model = sm.Logit(y_train, X_train_const).fit(disp=0)
coef_df = pd.DataFrame({
    'Variable': ['const'] + FEATURES,
    'Coeficiente': logit_model.params,
    'p-value': logit_model.pvalues,
    'Odds Ratio': np.exp(logit_model.params)
})

# ==================================================
# SECCIÓN 1: INTRODUCCIÓN CLÍNICA
# ==================================================
st.markdown("""
La detección temprana del cáncer de mama es un desafío clínico crítico.  
Este estudio utiliza el **dataset Wisconsin Diagnostic Breast Cancer (WDBC)** con 569 muestras de aspirado de aguja fina (FNA),  
donde se extrajeron 30 características morfológicas del núcleo celular.

Mediante un análisis de correlación y selección de variables (RFE), se redujo el modelo a **4 características clave** que capturan  
la variabilidad y los valores extremos (worst) del tumor. Se entrenó una **regresión logística** que alcanza una **precisión del 95.6%**  
y un **AUC de 0.9974**, demostrando una capacidad casi perfecta para distinguir entre lesiones benignas y malignas.

A continuación se muestra la **matriz de correlación** entre las variables seleccionadas y el diagnóstico, así como un **gráfico de dispersión**  
donde se visualiza la separación entre benignos y malignos, y se puede ubicar al paciente actual.
""")

# ==================================================
# SECCIÓN 2: ANÁLISIS EXPLORATORIO (MATRIZ DE CORRELACIÓN )
# ==================================================
st.subheader(" Análisis exploratorio de las variables clave")

st.markdown("**Matriz de correlación** (variables predictoras vs diagnóstico)")
corr_vars = FEATURES + ['target']
corr_matrix = df[corr_vars].corr()
fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                     color_continuous_scale="Blues", zmin=-1, zmax=1)
fig_corr.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=0))
fig_corr.update_xaxes(side="bottom")
st.plotly_chart(fig_corr, use_container_width=True, config=config_plotly)



# ==================================================
# SECCIÓN 3: ENTRADA DE DATOS Y PREDICCIÓN (lado a lado)
# ==================================================
st.markdown("---")
st.subheader("📋 Ingreso de parámetros celulares")

# Obtener rangos clínicos reales
ranges = {var: (df[var].min(), df[var].max()) for var in FEATURES}

col_form, col_result = st.columns([1, 1], gap="large")

with col_form:
    st.markdown("**Complete las siguientes mediciones obtenidas del FNA:**")
    area2 = st.number_input("Área (error estándar) – area2", 
                        min_value=float(ranges['area2'][0]), max_value=float(ranges['area2'][1]),
                        value=st.session_state.area2, step=1.0, format="%.2f",
                        key='area2')
    radius3 = st.number_input("Radio máximo (worst) – radius3",
                              min_value=float(ranges['radius3'][0]), max_value=float(ranges['radius3'][1]),
                              value=st.session_state.radius3, step=0.1, format="%.2f",
                              key='radius3')
    area3 = st.number_input("Área máxima (worst) – area3",
                            min_value=float(ranges['area3'][0]), max_value=float(ranges['area3'][1]),
                            value=st.session_state.area3, step=10.0, format="%.2f",
                            key='area3')
    concave_points3 = st.number_input("Puntos cóncavos máximos – concave_points3",
                                      min_value=float(ranges['concave_points3'][0]), max_value=float(ranges['concave_points3'][1]),
                                      value=st.session_state.concave_points3, step=0.005, format="%.4f",
                                      key='concave_points3')

    boton_pulsado = st.button(" Calcular diagnóstico", type="primary", use_container_width=True)
    if boton_pulsado:
        st.session_state.predecir = True
        

with col_result:
    st.markdown("**Resultado clínico**")
    if st.session_state.predecir:
        # Crear DataFrame de entrada
        input_df = pd.DataFrame([[area2, radius3, area3, concave_points3]], columns=FEATURES)
        # Aplicar misma transformación logarítmica a las variables que lo necesiten (según skew >1.5)
        skewness_input = df[FEATURES].skew()
        high_skew_vars = skewness_input[skewness_input > 1.5].index.tolist()
        input_transformed = input_df.copy()
        for var in high_skew_vars:
            if var in input_transformed.columns:
                input_transformed[var] = np.log1p(input_transformed[var])
        input_scaled = scaler.transform(input_transformed)
        prob_mal = model.predict_proba(input_scaled)[0][1]
        pred_class = model.predict(input_scaled)[0]
        
        # Mostrar diagnóstico con colores suaves
        if pred_class == 0:
            st.markdown("#### ✅ **BENIGNO**")
            st.markdown(f"*Probabilidad de malignidad:* **{prob_mal:.2%}**")
            gauge_color = "#2E86AB"  # azul médico
            texto_color = "normal"
        else:
            st.markdown("#### ⚠️ **MALIGNO**")
            st.markdown(f"*Probabilidad de malignidad:* **{prob_mal:.2%}**")
            gauge_color = "#D9534F"  # coral oscuro
            texto_color = "normal"
        
        # Gauge semicircular (half-donut)
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob_mal * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Confiabilidad del modelo"},
            gauge = {
                'axis': {'range': [0, 100], 'tickvals': [0, 20, 40, 60, 80, 100], 'ticktext': ['0', '20', '40', '60', '80', '100']},
                'bar': {'color': gauge_color, 'thickness': 0.3},
                'bgcolor': "white",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 30], 'color': "#e8f4f8"},
                    {'range': [30, 70], 'color': "#fff3e0"},
                    {'range': [70, 100], 'color': "#fde4e4"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 2},
                    'thickness': 0.4,
                    'value': 50
                }
            }
        ))
        fig_gauge.update_layout(
            height=320,
            margin=dict(t=40, b=30, l=40, r=40),
            font=dict(size=13),
            autosize=True
            )
        st.plotly_chart(fig_gauge, use_container_width=True, config=config_plotly)
        
        st.caption("El umbral de decisión se establece en 50% de probabilidad.")
        st.markdown("---")
        st.caption(" *Esta herramienta es de apoyo diagnóstico y no reemplaza el criterio médico.*")
    else:
        st.info(" Ingrese los valores y presione 'Calcular diagnóstico'.")
        # Mostrar un gráfico de gauge vacío o placeholder
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = 0,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Confiabilidad del modelo"},
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "lightgray", 'thickness': 0.3}}
        ))
        fig_gauge.update_layout(height=250, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_gauge, use_container_width=True, config=config_plotly)
# ==================================================
# SCATTER PLOT INTERACTIVO CON EL PUNTO DEL PACIENTE
# ==================================================
st.markdown("---")
st.subheader(" Ubicación del paciente en el espacio de características")

if st.session_state.predecir:
    # Añadir el punto del paciente al scatter plot existente
    fig_scatter_paciente = px.scatter(df, x='area3', y='concave_points3', color='Diagnostico',
                                      color_discrete_map={'B': '#2E86AB', 'M': '#D9534F'},
                                      title="", labels={'area3': 'Área máxima', 'concave_points3': 'Puntos cóncavos máximos'})
    fig_scatter_paciente.add_scatter(x=[st.session_state.area3], y=[st.session_state.concave_points3], mode='markers',
                                     marker=dict(size=18, color='black', symbol='x', line=dict(width=2, color='white')),
                                     name='Paciente actual')
    fig_scatter_paciente.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_scatter_paciente, use_container_width=True, config=config_plotly)
    st.caption("El punto negro con 'x' representa la muestra del paciente. Si se ubica cerca de la zona roja, la probabilidad de malignidad es alta.")
else:
    st.info("Complete el formulario y calcule el diagnóstico para visualizar la posición del paciente en el gráfico.")


# ==================================================
# SECCIÓN 4: ESTADÍSTICAS DEL MODELO (RIGOR CIENTÍFICO)
# ==================================================
st.markdown("---")
st.subheader(" Validación del modelo de regresión logística")

# Métricas en columnas
col_met1, col_met2, col_met3, col_met4, col_met5 = st.columns(5)
col_met1.metric("Accuracy", f"{acc:.3f}")
col_met2.metric("Sensibilidad", f"{sens:.3f}")
col_met3.metric("Especificidad", f"{spec:.3f}")
col_met4.metric("Precisión", f"{prec:.3f}")
col_met5.metric("F1-Score", f"{f1:.3f}")

# Matriz de confusión y curva ROC
col_cm, col_roc = st.columns(2)
with col_cm:
    st.markdown("**Matriz de confusión** (test set)")
    cm_df = pd.DataFrame(cm, index=["Real Benigno", "Real Maligno"], columns=["Pred. Benigno", "Pred. Maligno"])
    st.dataframe(cm_df, use_container_width=True)
with col_roc:
    st.markdown("**Curva ROC**")
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {roc_auc:.3f})', line=dict(color='#2E86AB', width=3)))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Aleatorio', line=dict(dash='dash', color='gray')))
    fig_roc.update_layout(xaxis_title="1 - Especificidad (FPR)", yaxis_title="Sensibilidad (TPR)", height=350, margin=dict(l=0, r=0, t=30, b=0))
    fig_roc.update_xaxes(range=[0,1])
    fig_roc.update_yaxes(range=[0,1])
    st.plotly_chart(fig_roc, use_container_width=True, config=config_plotly)

# Coeficientes con p-values
st.markdown("**Coeficientes del modelo y significancia estadística**")
st.dataframe(coef_df.style.format({'Coeficiente': '{:.4f}', 'p-value': '{:.4e}', 'Odds Ratio': '{:.4f}'}), use_container_width=True)

# Fórmula matemática correctamente renderizada
st.markdown("**Función de predicción**")
st.latex(r"""
P(\text{Maligno}) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3 + \beta_4 x_4)}}
""")
st.caption("Donde \(x_i\) son las características estandarizadas y transformadas, y \(\beta_i\) los coeficientes de la tabla anterior.")


# ==================================================
# PIE DE PÁGINA
# ==================================================
st.markdown("---")
st.caption("© 2026 OncoPredict – Desarrollado por Kevin Yael Oseguera Reyes | Basado en el dataset Wisconsin Diagnostic Breast Cancer (UW).")