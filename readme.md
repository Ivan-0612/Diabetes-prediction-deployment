# Sistema de Predicción de Diabetes

Una aplicación web interactiva construida con **Streamlit** y **Machine Learning** que predice la probabilidad de que un paciente padezca diabetes basándose en datos demográficos, hábitos y métricas médicas.

## Características

* **Formulario Interactivo:** Entrada de datos sencilla para el usuario (edad, IMC, glucosa, etc.).
* **Predicción en Tiempo Real:** Utiliza un modelo de Regresión Logística entrenado para calcular el riesgo al instante.
* **Explicabilidad (XAI):** Muestra gráficos de **SHAP** para explicar qué variables aumentaron o disminuyeron el riesgo del paciente específico.

## Estructura del Proyecto

```text
├── app.py                # Código principal de la aplicación Streamlit
├── deploy_model.pkl      # Modelo entrenado (Pipeline + Regresión Logística)
├── requirements.txt      # Lista de librerías necesarias
└── README.md             # Documentación del proyecto

```

El proyecto esta desplegado en Render: https://diabetes-prediction-rlji.onrender.com/
