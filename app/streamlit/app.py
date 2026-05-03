from pathlib import Path

import folium
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium


st.set_page_config(
    page_title="Predicción de Calidad del Aire",
    page_icon="🌍",
    layout="wide",
)


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "air_weather_features.csv"

API_URL = "http://localhost:8006/predict"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(["station", "time"]).reset_index(drop=True)
    return df


def get_aqi_color(aqi):
    if aqi <= 50:
        return "green"
    elif aqi <= 100:
        return "orange"
    return "red"


def get_aqi_message(aqi):
    if aqi <= 50:
        return "Calidad del aire buena"
    elif aqi <= 100:
        return "Calidad del aire moderada"
    return "Alerta: calidad del aire mala"


df = load_data()

st.title("🌍 Predicción de Calidad del Aire Urbano")
st.write(
    "Dashboard interactivo conectado a FastAPI para visualizar datos y predecir el AQI futuro por estación."
)

st.sidebar.header("Filtros")

stations = sorted(df["station"].unique())
selected_station = st.sidebar.selectbox("Selecciona estación", stations)

contaminants = ["pm10", "pm2_5", "no2", "ozone", "so2", "co"]
selected_pollutant = st.sidebar.selectbox("Selecciona contaminante", contaminants)

df_station = df[df["station"] == selected_station].copy()
latest = df_station.sort_values("time").iloc[-1]

col1, col2, col3, col4 = st.columns(4)

col1.metric("AQI actual", round(latest["aqi"], 2))
col2.metric("PM10", round(latest["pm10"], 2))
col3.metric("NO₂", round(latest["no2"], 2))
col4.metric("O₃", round(latest["ozone"], 2))

if latest["aqi"] <= 50:
    st.success(get_aqi_message(latest["aqi"]))
elif latest["aqi"] <= 100:
    st.warning(get_aqi_message(latest["aqi"]))
else:
    st.error(get_aqi_message(latest["aqi"]))


st.subheader("🔮 Predicción futura del AQI vía FastAPI")

st.write(
    "La predicción se obtiene llamando al endpoint `/predict` de la API FastAPI."
)

if "api_prediction" not in st.session_state:
    st.session_state.api_prediction = None

if st.button("Predecir AQI futuro"):
    try:
        response = requests.post(
            API_URL,
            json={"station": selected_station},
            timeout=10,
        )

        if response.status_code == 200:
            st.session_state.api_prediction = response.json()
        else:
            st.error(f"Error en la API: {response.status_code}")
            st.write(response.text)

    except requests.exceptions.ConnectionError:
        st.error(
            "No se pudo conectar con la API. "
            "Asegúrate de ejecutar FastAPI en el puerto 8000."
        )

    except Exception as e:
        st.error(f"Error inesperado: {e}")

if st.session_state.api_prediction is not None:
    data = st.session_state.api_prediction

    col_a, col_b = st.columns(2)

    col_a.metric("AQI actual", data["current_aqi"])
    col_b.metric(
        "AQI predicho próxima hora",
        data["predicted_aqi_next_hour"],
    )

    alert = data["alert"]

    if alert["level"] == "buena":
        st.success(alert["message"])
    elif alert["level"] == "moderada":
        st.warning(alert["message"])
    else:
        st.error(alert["message"])

    with st.expander("Ver respuesta completa de la API"):
        st.json(data)

st.subheader(f"📈 Evolución temporal del AQI - {selected_station}")
st.line_chart(df_station.set_index("time")[["aqi"]])

st.subheader(
    f"🌫️ Evolución temporal de {selected_pollutant.upper()} - {selected_station}"
)
st.line_chart(df_station.set_index("time")[[selected_pollutant]])

st.subheader("📊 Comparativa AQI entre estaciones")

df_compare = df.pivot_table(
    index="time",
    columns="station",
    values="aqi",
)

st.line_chart(df_compare)

st.subheader("🗺️ Mapa de estaciones")

latest_by_station = (
    df.sort_values("time")
    .groupby("station")
    .tail(1)
    .reset_index(drop=True)
)

map_center = [
    latest_by_station["latitude"].mean(),
    latest_by_station["longitude"].mean(),
]

m = folium.Map(location=map_center, zoom_start=6)

for _, row in latest_by_station.iterrows():
    color = get_aqi_color(row["aqi"])

    popup_text = f"""
    <b>{row['station']}</b><br>
    AQI actual: {row['aqi']:.2f}<br>
    PM10: {row['pm10']:.2f}<br>
    PM2.5: {row['pm2_5']:.2f}<br>
    NO₂: {row['no2']:.2f}<br>
    O₃: {row['ozone']:.2f}
    """

    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=10,
        popup=popup_text,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.75,
    ).add_to(m)

st_folium(m, width=900, height=500)

st.subheader("🏆 Ranking de estaciones por AQI")

ranking = latest_by_station.sort_values("aqi", ascending=False)

st.dataframe(
    ranking[
        [
            "station",
            "aqi",
            "pm10",
            "pm2_5",
            "no2",
            "ozone",
            "temperature",
            "precipitation",
        ]
    ],
    use_container_width=True,
)

st.subheader("📋 Datos recientes de la estación seleccionada")

st.dataframe(
    df_station.sort_values("time", ascending=False).head(20),
    use_container_width=True,
)