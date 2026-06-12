import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuración para que los gráficos se muestren en el navegador
import plotly.io as pio
pio.renderers.default = 'browser'  # Abre en navegador automáticamente

# Cargar datos
df = pd.read_csv('wdbc_dataset.csv')

print("=== ANÁLISIS EXPLORATORIO DE DATOS (EDA) ===")
print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
print(f"Columnas: {df.columns.tolist()}\n")

# 1. Estadísticos descriptivos básicos
print("--- Estadísticos descriptivos (primeras 5 variables) ---")
print(df.iloc[:, 2:7].describe())
print("\n")

# 2. Balance de clases
print("--- Distribución de Diagnóstico ---")
diagnostico_counts = df['Diagnostico'].value_counts().reset_index()
diagnostico_counts.columns = ['Diagnostico', 'Cantidad']
print(df['Diagnostico'].value_counts())
print(f"Proporción: {df['Diagnostico'].value_counts(normalize=True)}\n")

# Gráfico 1: Distribución de clases (barras interactivo)
fig1 = px.bar(diagnostico_counts, x='Diagnostico', y='Cantidad',
              title='Distribución de Casos: Benigno (B) vs Maligno (M)',
              text='Cantidad', color='Diagnostico',
              color_discrete_map={'B': '#2E86AB', 'M': '#A23B72'})
fig1.update_traces(textposition='outside')
fig1.update_layout(width=600, height=500)
fig1.show()
fig1.write_html("output_distribucion_clases.html")

# 3. Matriz de correlación (solo variables numéricas, excluyendo ID)
numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('ID')
corr_matrix = df[numeric_cols].corr()

# Heatmap interactivo
fig2 = px.imshow(corr_matrix,
                 text_auto=True,
                 aspect="auto",
                 color_continuous_scale='RdBu_r',
                 title='Matriz de Correlación - Todas las variables numéricas',
                 width=1000, height=900)
fig2.update_layout(font=dict(size=8))
fig2.show()
fig2.write_html("output_matriz_correlacion.html")

# 4. Boxplots  para outliers (usando una figura con subplots)
key_vars = ['area1', 'radius1', 'concavity1', 'area3', 'concave_points3']
fig3 = make_subplots(rows=2, cols=3,
                     subplot_titles=key_vars,
                     shared_yaxes=False)

for i, var in enumerate(key_vars):
    row = i // 3 + 1
    col = i % 3 + 1
    fig3.add_trace(go.Box(y=df[var], name=var, boxmean='sd'), row=row, col=col)

fig3.update_layout(height=800, width=1200, title_text="Boxplots para detección de outliers")
fig3.show()
fig3.write_html("output_boxplots.html")

# 5. Histogramas con densidad (distribuciones)
fig4 = make_subplots(rows=2, cols=3,
                     subplot_titles=key_vars,
                     shared_yaxes=False)

for i, var in enumerate(key_vars):
    row = i // 3 + 1
    col = i % 3 + 1
    hist_data = df[var]
    fig4.add_trace(go.Histogram(x=hist_data, nbinsx=30, name=var,
                                histnorm='probability density', opacity=0.7), row=row, col=col)
    # Añadir curva de densidad (KDE aproximado)
    fig4.add_trace(go.Histogram(x=hist_data, nbinsx=30, histnorm='probability density',
                                cumulative_enabled=False, opacity=0, showlegend=False), row=row, col=col)

fig4.update_layout(height=800, width=1200, title_text="Distribución de variables clave")
fig4.show()
fig4.write_html("output_histogramas.html")

# 6. Gráfico: relación entre area1 y concavity1 coloreado por diagnóstico
fig5 = px.scatter(df, x='area1', y='concavity1', color='Diagnostico',
                  title='Relación entre Área y Concavidad (por diagnóstico)',
                  labels={'area1': 'Área media', 'concavity1': 'Concavidad media'},
                  color_discrete_map={'B': '#2E86AB', 'M': '#A23B72'},
                  hover_data=['radius1', 'texture1'])
fig5.update_layout(width=800, height=600)
fig5.show()
fig5.write_html("output_scatter_area_concavity.html")

print("\n EDA completado. Se han generado los siguientes archivos HTML :")
print(" - output_distribucion_clases.html")
print(" - output_matriz_correlacion.html")
print(" - output_boxplots.html")
print(" - output_histogramas.html")
print(" - output_scatter_area_concavity.html")
