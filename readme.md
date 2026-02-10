# 🏥 Sistema de Predicción de Diabetes

Una aplicación web interactiva construida con **Streamlit** y **Machine Learning** que predice la probabilidad de que un paciente padezca diabetes basándose en datos demográficos, hábitos y métricas médicas.

## 🚀 Características

* **Formulario Interactivo:** Entrada de datos sencilla para el usuario (edad, IMC, glucosa, etc.).
* **Predicción en Tiempo Real:** Utiliza un modelo de Regresión Logística entrenado para calcular el riesgo al instante.
* **Explicabilidad (XAI):** Muestra gráficos de **SHAP** para explicar qué variables aumentaron o disminuyeron el riesgo del paciente específico.

## 📋 Requisitos Previos

Asegúrate de tener instalado **Python 3.8** o superior.

Las librerías necesarias son:

* streamlit
* pandas
* numpy
* joblib
* shap
* matplotlib
* scikit-learn

## 🛠️ Instalación y Uso

1. **Clona o descarga este repositorio** en tu ordenador.
2. **Instala las dependencias** (Se recomienda usar un entorno virtual):
```bash
pip install -r requirements.txt

```


3. **Asegúrate de tener el modelo:**
El archivo `deploy_model.pkl` debe estar en la misma carpeta que `app.py`.
4. **Ejecuta la aplicación:**
```bash
streamlit run app.py

```

5. La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`.

## 📂 Estructura del Proyecto

```text
├── app.py                # Código principal de la aplicación Streamlit
├── deploy_model.pkl      # Modelo entrenado (Pipeline + Regresión Logística)
├── requirements.txt      # Lista de librerías necesarias
└── README.md             # Documentación del proyecto

```