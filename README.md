# 🔬 OncoPredict · Predicción de Cáncer de Mama con Regresión Logística

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2-orange)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)](https://streamlit.io/)
[![AUC](https://img.shields.io/badge/AUC-0.9974-brightgreen)]()

## 📌 Descripción del proyecto

**OncoPredict** es una herramienta de apoyo al diagnóstico clínico basada en un modelo de **regresión logística** que predice si un tumor de mama es **benigno** o **maligno** a partir de cuatro características morfológicas extraídas de imágenes de aspirado de aguja fina (FNA).

El modelo fue entrenado con el dataset **Wisconsin Diagnostic Breast Cancer (WDBC)** que contiene 569 muestras y 30 atributos. Mediante un proceso de selección de variables (RFE) se identificaron las cuatro características más discriminantes:

- `area2`  (error estándar del área)
- `radius3` (valor máximo del radio – *worst*)
- `area3`   (valor máximo del área – *worst*)
- `concave_points3` (valor máximo de puntos cóncavos – *worst*)

El modelo alcanza una **precisión del 95.6%** y un **AUC de 0.9974**, lo que indica una capacidad casi perfecta para distinguir entre lesiones benignas y malignas.

##  Resultados clave (validación en test set)

| Métrica | Valor |
|---------|-------|
| **Accuracy** | 95.6% |
| **AUC** | 0.9974 |
| **Sensibilidad (Recall)** | 92.9% (39/42 malignos detectados) |
| **Especificidad** | 97.2% (70/72 benignos detectados) |
| **Precisión** | 95.1% |
| **F1‑Score** | 0.940 |
| **Pseudo R² de McFadden** | 0.814 |

##  Interpretación clínica

> *“Las variables seleccionadas por el modelo son medidas de dispersión o valores extremos (‘worst’). Esto indica que lo que más distingue la malignidad no es el valor típico del tumor, sino **cuán extremo puede llegar a ser** y su **variabilidad**. Los tumores malignos tienden a presentar mayor tamaño, formas más irregulares y valores atípicos más extremos.”*

El modelo es **interpretable** y proporciona una probabilidad de malignidad, lo que ayuda al clínico a tomar decisiones graduales.

##  Tecnologías utilizadas

- **Python** (pandas, numpy, scikit-learn, statsmodels, plotly)
- **Regresión Logística** con selección de variables RFE
- **Visualización interactiva** con Plotly
- **Aplicación web** con Streamlit

##  Estructura del repositorio
```text
breast-cancer-predictor/
├── app.py # Aplicación interactiva con Streamlit
├── 03_modelado_logistico.py # Entrenamiento y selección de variables
├── modelo_final.pkl # Modelo serializado
├── scaler_final.pkl # Escalador (StandardScaler)
├── variables_finales.txt # Lista de las 4 variables seleccionadas
├── assets/ # Imágenes (opcional)
├── requirements.txt # Dependencias
└── README.md
```

##  Cómo ejecutar localmente

1. Clona el repositorio:
   ```bash
   git clone https://github.com/koseguera-r/breast-cancer-predictor.git
   cd breast-cancer-predictor

    python -m venv venv
    source venv/bin/activate        # Linux/Mac
    pip install -r requirements.txt
    streamlit run app.py
```
``` Referencias

    Dataset: Wisconsin Diagnostic Breast Cancer (WDBC) – UCI Machine Learning Repository

    Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1995). Machine learning techniques to diagnose breast cancer from fine‑needle aspirates.```



##👤 Autor

**Kevin Yael Oseguera Reyes**  
[kevinose1666@gmail.com](mailto:kevinose1666@gmail.com)  
[LinkedIn](https://www.linkedin.com/in/-osegukevin3b0) | [GitHub](https://github.com/koseguera-r)

---
 *Si este proyecto te ha sido útil, no olvides darle una estrella.*