import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import joblib
from shapely.geometry import Point
from huggingface_hub import hf_hub_download, login
from datetime import datetime
from imdlib import get_data
import os
from sklearn.preprocessing import LabelEncoder

# Optional: set this as a default if env variable isn't set
HF_TOKEN = os.getenv("HF_TOKEN")


@st.cache_resource
def load_districts():
    path = os.path.join("data", "gadm41_IND_2.json.zip")
    districts = gpd.read_file(path)
    districts = districts.rename(columns={"NAME_1": "STATE", "NAME_2": "DISTRICT"})
    return districts

@st.cache_resource
def load_encoder():
    path = hf_hub_download(
        repo_id="BindhuGrashia/flood_region_forecast-v2",
        filename="region_encoder.joblib",
        repo_type="model",
        token=HF_TOKEN
    )
    return joblib.load(path)

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="BindhuGrashia/flood_region_forecast-v2",
        filename="flood_region_forecast-v2.pkl",
        token=HF_TOKEN
    )
    return joblib.load(model_path)

def get_district_from_latlon(lat, lon, districts):
    point = Point(lon, lat)
    for _, row in districts.iterrows():
        if row['geometry'].contains(point):
            return row['STATE'], row['DISTRICT']
    return None, None

def get_rainfall_features(lat, lon):
    try:
        lat = float(lat)
        lon = float(lon)

        districts = load_districts()

        end_year = datetime.now().year - 1
        start_year = end_year - 1

        os.makedirs("yearly_district_rainfall", exist_ok=True)

        dfs = []
        for year in range(start_year, end_year + 1):
            data = get_data('rain', year, year, fn_format='yearwise')
            ds = data.get_xarray()
            df = ds.to_dataframe().reset_index().dropna(subset=['rain'])

            geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
            gdf_rain = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

            gdf_joined = gpd.sjoin(
                gdf_rain,
                districts[['geometry', 'STATE', 'DISTRICT']],
                how='left',
                predicate='within'
            )

            gdf_joined['year'] = year
            dfs.append(gdf_joined)

        df_all = pd.concat(dfs).dropna(subset=['DISTRICT'])
        df_all = df_all.rename(columns={'rain': 'RAINFALL'})
        df_all['RAINFALL'] = df_all['RAINFALL'].replace(-999, np.nan)
        df_all = df_all.dropna(subset=['RAINFALL'])
        df_all['REGION'] = df_all['STATE'] + "_" + df_all['DISTRICT']

        le = load_encoder()
        df_all['REGION_CODE'] = le.transform(df_all['REGION'])
        df_all['DATE'] = pd.to_datetime(df_all['time'])

        rolled = (
            df_all
            .sort_values(['REGION_CODE', 'DATE'])
            .groupby('REGION_CODE')
            .rolling('730D', on='DATE')[['RAINFALL']]
            .agg(['mean', 'max', 'std', 'sum', 'count'])
            .reset_index()
        )

        rolled.columns = [
            'REGION_CODE', 'DATE',
            'rain_mean', 'rain_max', 'rain_std', 'rain_sum', 'rain_count'
        ]

        rolled = rolled.dropna(subset=['rain_std']).reset_index(drop=True)

        state, district = get_district_from_latlon(lat, lon, districts)
        region = f"{state}_{district}"
        region_code = int(le.transform([region])[0])

        latest_features = rolled[rolled['REGION_CODE'] == region_code].sort_values("DATE").iloc[-1]
        features = latest_features.drop(labels=["DATE"]).to_dict()
        features = {k: float(v) for k, v in features.items()}

        return features

    except Exception as e:
        st.warning(f"⚠️ Could not extract rainfall features: {e}")
        return None

def predict_flood(lat, lon):
    districts = load_districts()
    model = load_model()

    state, district = get_district_from_latlon(lat, lon, districts)
    if not district or not state:
        return "❌ District or state not found."

    region = f"{state}_{district}"

    try:
        le = load_encoder()
        region_code = int(le.transform([region])[0])
    except Exception as e:
        return f"⚠️ Could not encode region '{region}': {e}"

    features = get_rainfall_features(lat, lon)
    if not features:
        return "⚠️ Could not extract rainfall features."

    features["REGION_CODE"] = region_code
    X = pd.DataFrame([features]).astype(float)
    prediction = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    return {
        "district": district,
        "state": state,
        "prediction": "🌊 Flood Likely" if prediction == 1 else "✅ No Flood",
        "confidence": round(prob, 3),
        "features": features
    }

# UI Layout
st.set_page_config(page_title="Flood Risk Predictor", layout="centered")
st.title("🌧️ Flood Risk Predictor (Real IMD Data)")
st.markdown("Enter a **latitude** and **longitude** to estimate flood risk for next year (based on real IMD data):")

lat = st.number_input("Latitude", value=11.2588, format="%.4f")
lon = st.number_input("Longitude", value=75.7804, format="%.4f")

if st.button("🔍 Predict Flood Risk"):
    with st.spinner("Fetching IMD rainfall data & predicting..."):
        result = predict_flood(lat, lon)
        if isinstance(result, dict):
            st.success(f"📍 District: `{result['district']}`")
            st.markdown(f"### 🔮 Prediction: {result['prediction']}")
            st.json(result['features'])
        else:
            st.error(result)

