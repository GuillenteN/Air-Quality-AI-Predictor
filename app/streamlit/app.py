from pathlib import Path

import folium
import joblib
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium


st.set_page_config(
    page_title="Predicción de Calidad del Aire",
    page_icon="🌍",
    layout="wide",
)


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "air_weather_features.csv"

MODEL_PATH = BASE_DIR / "models" / "xgb_future_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "future_scaler.pkl"
FEATURES_PATH = BASE_DIR / "models" / "future_features.pkl"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(["station", "time"]).reset_index(drop=True)
    return df


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists() or not FEATURES_PATH.exists():
        return None, None, None

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)

    return model, scaler, features


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
model, scaler, features = load_model()

st.title("🌍 Predicción de Calidad del Aire Urbano")
st.write("Dashboard interactivo para visualizar y predecir el AQI futuro por estación.")

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

st.subheader("🔮 Predicción futura del AQI")

if model is None:
    st.info(
        "No se encontró el modelo futuro. "
        "Primero guarda `xgb_future_model.pkl`, `future_scaler.pkl` y `future_features.pkl`."
    )
else:
    X_input = latest[features].to_frame().T
    X_input_scaled = scaler.transform(X_input)

    predicted_aqi = model.predict(X_input_scaled)[0]

    col_a, col_b = st.columns(2)

    col_a.metric("AQI actual", round(latest["aqi"], 2))
    col_b.metric("AQI predicho próxima hora", round(predicted_aqi, 2))

    if predicted_aqi <= 50:
        st.success(f"Predicción: {get_aqi_message(predicted_aqi)}")
    elif predicted_aqi <= 100:
        st.warning(f"Predicción: {get_aqi_message(predicted_aqi)}")
    else:
        st.error(f"Predicción: {get_aqi_message(predicted_aqi)}")

st.subheader(f"📈 Evolución temporal del AQI - {selected_station}")
st.line_chart(df_station.set_index("time")[["aqi"]])

st.subheader(f"🌫️ Evolución temporal de {selected_pollutant.upper()} - {selected_station}")
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