import streamlit as st
import requests
import numpy as np
import pandas as pd

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Predicción NO2",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Predicción de Calidad del Aire")
st.markdown("### Predicción de NO₂ a 24 horas con Machine Learning")
st.write(
    "Introduce los valores actuales y el sistema estimará el nivel de NO₂ para las próximas 24 horas."
)

st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("## 🧾 Datos de entrada")

    no2 = st.number_input("NO₂ actual", min_value=0.0, value=50.0, step=1.0)
    no2_lag_1 = st.number_input("NO₂ hace 1 hora", min_value=0.0, value=45.0, step=1.0)
    no2_lag_24 = st.number_input("NO₂ hace 24 horas", min_value=0.0, value=55.0, step=1.0)

with col2:
    st.markdown("## ⏱️ Información temporal")

    hour = st.slider("Hora del día", 0, 23, 12)
    day_of_week = st.slider("Día de la semana (0=Lunes, 6=Domingo)", 0, 6, 2)
    month = st.slider("Mes", 1, 12, 1)

is_weekend = 1 if day_of_week >= 5 else 0
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)

st.info(
    f"📅 Día seleccionado: {'fin de semana' if is_weekend else 'día laborable'} | "
    f"Hora: {hour}:00"
)

st.divider()

if st.button("🔮 Predecir calidad del aire", use_container_width=True):
    payload = {
        "no2": no2,
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": is_weekend,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "no2_lag_1": no2_lag_1,
        "no2_lag_24": no2_lag_24
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)

        if response.status_code == 200:
            result = response.json()
            prediction = result["prediction_no2_24h"]

            st.markdown("## 📌 Resultado de la predicción")

            m1, m2, m3 = st.columns(3)

            with m1:
                st.metric("NO₂ actual", f"{no2:.2f}")

            with m2:
                st.metric("Predicción NO₂ a 24h", f"{prediction:.2f}")

            with m3:
                difference = prediction - no2
                st.metric("Variación estimada", f"{difference:.2f}")

            if prediction < 50:
                st.success("🟢 Calidad del aire buena")
                level = "Buena"
            elif prediction < 100:
                st.warning("🟡 Calidad del aire moderada")
                level = "Moderada"
            else:
                st.error("🔴 Alta contaminación")
                level = "Alta contaminación"

            st.markdown("## 📈 Evolución reciente y predicción")

            chart_df = pd.DataFrame({
                "NO₂": [no2_lag_24, no2_lag_1, no2, prediction]
            }, index=["Hace 24h", "Hace 1h", "Actual", "Predicción 24h"])

            st.line_chart(chart_df)

            st.markdown("## 🧠 Interpretación")

            if difference > 0:
                st.write(
                    f"El modelo estima una **subida** del NO₂ de aproximadamente "
                    f"**{difference:.2f} unidades** en las próximas 24 horas."
                )
            elif difference < 0:
                st.write(
                    f"El modelo estima una **bajada** del NO₂ de aproximadamente "
                    f"**{abs(difference):.2f} unidades** en las próximas 24 horas."
                )
            else:
                st.write("El modelo estima que el nivel de NO₂ se mantendrá estable.")

            st.caption(
                "Modelo utilizado: XGBoost entrenado con datos históricos de calidad del aire."
            )

        else:
            st.error("❌ Error en la API")
            st.write(response.text)

    except Exception as e:
        st.error("❌ No se pudo conectar con la API")
        st.write("Comprueba que FastAPI está ejecutándose en el puerto 8000.")
        st.code("python -m uvicorn app.api.main:app --reload")
        st.write(e)