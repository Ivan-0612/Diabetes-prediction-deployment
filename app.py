import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="Predicción de diabetes en pacientes", layout="wide")


# Cargar modelo
@st.cache_resource
def load_model():
    return joblib.load("deploy_model.pkl")


try:
    model = load_model()
except Exception as e:
    st.error("No se pudo cargar el modelo")
    with st.expander("Ver detalles del error"):
        st.code(e)

# Inputs del modelo
# todos los valores por defecto son la mediana de cada variable en el dataset, para que el usuario tenga una referencia
st.title("Sistema de predicción de diabetes")
st.markdown(
    """
Esta aplicación utiliza un modelo de **Machine Learning (Regresión Logística)** para predecir la probabilidad de que un paciente padezca diabetes.
"""
)

with st.form("formulario_inputs"):
    st.subheader("Ingresa los datos del paciente")
    tab1, tab2 = st.tabs(["Datos personales", "Datos médicos"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            gender = st.radio("Género", ["Female", "Male", "Other"], horizontal=True)
            ethnicity = st.radio(
                "Etnia",
                ["White", "Hispanic", "Black", "Asian", "Other"],
                horizontal=True,
            )
            employment_status = st.radio(
                "Situación laboral",
                ["Employed", "Retired", "Unemployed", "Student"],
                horizontal=True,
            )
            smoking_status = st.radio(
                "Condición de fumador", ["Never", "Current", "Former"], horizontal=True
            )
            income_level = st.radio(
                "Nivel de ingresos",
                ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"],
                horizontal=True,
            )
        with col2:
            education_level = st.radio(
                "Nivel educativo",
                ["Highschool", "Graduate", "Postgraduate", "No formal"],
                horizontal=True,
            )
            age = st.slider("Edad", 18, 90, 50)
            alcohol_consumption_per_week = st.slider(
                "Número de bebidas alcohólicas consumidas semanalmente", 0, 30, 2
            )
            physical_activity_minutes_per_week = st.slider(
                "Actividad física realizada semanalmente (minutos)", 0, 600, 100
            )

    with tab2:
        col3, col4 = st.columns(2)
        with col3:
            diet_score = st.slider("Puntuación de la dieta", 0, 10, 6)
            screen_time_hours_per_day = st.slider(
                "Número de horas diarias frente a pantalla", 0, 12, 6
            )
            sleep_hours_per_day = st.slider(
                "Número de horas de sueño diarias", 0, 12, 7
            )
            bmi = st.number_input("BMI", 15.0, 45.0, 25.6)
            waist_to_hip_ratio = st.number_input(
                "Relación cintura-cadera", 0.7, 1.2, 0.86
            )
            systolic_bp = st.number_input("Presión arterial sistólica", 90, 200, 120)
            diastolic_bp = st.number_input("Presión arterial diastólica", 60, 120, 75)

        with col4:
            heart_rate = st.number_input("Frecuencia cardíaca", 50, 120, 70)
            hdl_cholesterol = st.number_input("Colesterol HDL", 20, 100, 54)
            ldl_cholesterol = st.number_input("Colesterol LDL", 50, 200, 102)
            triglycerides = st.number_input("Triglicéridos", 50, 500, 150)

            family_history_diabetes = st.checkbox("Antecedentes familiares de diabetes")
            cardiovascular_history = st.checkbox("Antecedentes cardiovasculares")
            hypertension_history = st.checkbox("Historial de hipertensión")

    submit_button = st.form_submit_button("Guardar datos y calcular riesgo")

    if submit_button:
        datos = {
            "age": age,
            "alcohol_consumption_per_week": alcohol_consumption_per_week,
            "physical_activity_minutes_per_week": physical_activity_minutes_per_week,
            "diet_score": diet_score,
            "sleep_hours_per_day": sleep_hours_per_day,
            "screen_time_hours_per_day": screen_time_hours_per_day,
            "family_history_diabetes": 1 if family_history_diabetes else 0,
            "hypertension_history": 1 if hypertension_history else 0,
            "cardiovascular_history": 1 if cardiovascular_history else 0,
            "bmi": bmi,
            "waist_to_hip_ratio": waist_to_hip_ratio,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "heart_rate": heart_rate,
            "hdl_cholesterol": hdl_cholesterol,
            "ldl_cholesterol": ldl_cholesterol,
            "triglycerides": triglycerides,
            "education_level": education_level,
            "income_level": income_level,
            "smoking_status": smoking_status,
            "employment_status": employment_status,
            "gender": gender,
            "ethnicity": ethnicity,
        }

        # Asegurar el orden de las columnas para coincidir con el entrenamiento
        df_final = pd.DataFrame([datos])

        with st.spinner("Analizando datos del paciente..."):
            # Predicción
            prediction = model.predict(df_final)[0]
            probability = model.predict_proba(df_final)[0][1]

            # Visualización
            st.markdown(
                f"El paciente tiene un **{probability:.1%}** de probabilidad de padecer diabetes."
            )
            st.progress(float(probability))

        # Explicabilidad
        st.subheader("Influencia de las Variables")
        st.write("Qué variables están afectando a la probabilidad dada por el modelo")

        # Valores shap
        # 1. Extraer los componentes del pipeline para aplicarlos a los datos
        pipeline_step = model.named_steps["preprocessor"]
        classifier_step = model.named_steps["classifier"]

        # B. Transformar los datos de entrada
        X_transformed = pipeline_step.transform(df_final)

        # C. Obtener nombres de columnas
        feature_names = pipeline_step.get_feature_names_out()

        # D. Crear DataFrame procesado
        X_processed_df = pd.DataFrame(X_transformed, columns=feature_names)

        # Eliminar el prefijo 'num_' de los nombres de las columnas
        X_processed_df.columns = [
            col.replace("num__", "") for col in X_processed_df.columns
        ]

        # E. Calcular valores shap
        masker = np.zeros((1, X_transformed.shape[1]))
        # Al no tener los datos de entrenamiento (X_train) aquí, uso una matriz de ceros
        explainer = shap.LinearExplainer(classifier_step, masker)
        shap_values = explainer(X_processed_df)

        # F. Crear el gráfico Waterfall
        # El gráfico muestra cómo pasamos de la probabilidad base a la final
        fig, ax = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(shap_values[0], max_display=8, show=False)

        plt.title("Impacto por variable")
        st.pyplot(fig)
        st.markdown(
            """
            **¿Cómo leer este gráfico?**
            * **Eje horizontal (E(f(x))):** Es la probabilidad base.
            * **Bloques Rojos (Derecha):** Factores que **aumentan** la probabilidad.
            * **Bloques Azules (Izquierda):** Factores que **disminuyen** la probabilidad. 
            * **Valores numéricos:** Indican cuánto aporta cada característica.
            """
        )
