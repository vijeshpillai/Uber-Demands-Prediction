import joblib
import pandas as pd
import streamlit as st
from pathlib import Path
import datetime as dt
from sklearn.pipeline import Pipeline
from sklearn import set_config
from time import sleep

set_config(transform_output="pandas")

# set the root path
root_path = Path(__file__).parent

# paths to data
plot_data_path = root_path / "data/external/plot_data.csv"
data_path = root_path / "data/processed/test.csv"

# paths to models
kmeans_path = root_path / "models/mb_kmeans.joblib"
scaler_path = root_path / "models/scaler.joblib"
encoder_path = root_path / "models/encoder.joblib"
model_path = root_path / "models/model.joblib"

# load objects
scaler = joblib.load(scaler_path)
encoder = joblib.load(encoder_path)
model = joblib.load(model_path)
kmeans = joblib.load(kmeans_path)

# load datasets
df_plot = pd.read_csv(plot_data_path)
df = pd.read_csv(data_path, parse_dates=["tpep_pickup_datetime"]).set_index("tpep_pickup_datetime")

# Streamlit UI
st.title("Uber Demand in New York City 🚕🌆")

st.sidebar.title("Options")
map_type = st.sidebar.radio("Select the type of Map",
                            ["Complete NYC Map", "Only for Neighborhood Regions"],
                            index=1)

st.subheader("Date")
date = st.date_input("Select the date", value=None,
                     min_value=dt.date(2016,3,1),
                     max_value=dt.date(2016,3,31))
st.write("**Date:**", date)

st.subheader("Time")
time = st.time_input("Select the time", value=None)
st.write("**Current Time:**", time)

if date and time:
    delta = dt.timedelta(minutes=15)
    next_interval = dt.datetime.combine(date, time) + delta
    st.write("Demand for Time: ", next_interval.time())

    index = pd.Timestamp(f"{date} {next_interval.time()}")
    st.write("**Date & Time:**", index)

    # Check if timestamp exists in df
    if index not in df.index:
        st.error(f"No data available for timestamp: {index}. Please select a different date/time.")
        st.stop()

    st.subheader("Location")
    sample_loc = df_plot.sample(1).reset_index(drop=True)
    lat = sample_loc["pickup_latitude"].item()
    long = sample_loc["pickup_longitude"].item()
    region = sample_loc["region"].item()
    st.write("**Your Current Location**")
    st.write(f"Lat: {lat}")
    st.write(f"Long: {long}")

    with st.spinner("Fetching your Current Region"):
        sleep(3)

    st.write("Region ID: ", region)

    scaled_cord = scaler.transform(sample_loc.iloc[:, 0:2])

    # Map colors
    colors = ["#FF0000", "#FF4500", "#FF8C00", "#FFD700", "#ADFF2F", 
              "#32CD32", "#008000", "#006400", "#00FF00", "#7CFC00", 
              "#00FA9A", "#00FFFF", "#40E0D0", "#4682B4", "#1E90FF", 
              "#0000FF", "#0000CD", "#8A2BE2", "#9932CC", "#BA55D3", 
              "#FF00FF", "#FF1493", "#C71585", "#FF4500", "#FF6347", 
              "#FFA07A", "#FFDAB9", "#FFE4B5", "#F5DEB3", "#EEE8AA"]
    region_colors = {region: colors[i] for i, region in enumerate(df_plot["region"].unique().tolist())}
    df_plot["color"] = df_plot["region"].map(region_colors)

    pipe = Pipeline([('encoder', encoder), ('reg', model)])

    if map_type == "Complete NYC Map":
        progress_bar = st.progress(0, text="Operation in progress. Please wait.")
        for percent in range(100):
            sleep(0.05)
            progress_bar.progress(percent + 1, text="Operation in progress. Please wait.")
        st.map(data=df_plot, latitude="pickup_latitude", longitude="pickup_longitude", size=0.01, color="color")
        progress_bar.empty()

        input_data = df.loc[index, :].sort_values("region")
        predictions = pipe.predict(input_data.drop(columns=["total_pickups"]))

        st.markdown("### Map Legend")
        for ind in range(0, 30):
            color = colors[ind]
            demand = predictions[ind]
            region_id = f"{ind} (Current region)" if region == ind else ind
            st.markdown(f'<div style="display: flex; align-items: center;">'
                        f'<div style="background-color:{color}; width: 20px; height: 10px; margin-right: 10px;"></div>'
                        f'Region ID: {region_id} <br>'
                        f"Demand: {int(demand)} <br> <br>", unsafe_allow_html=True)

    elif map_type == "Only for Neighborhood Regions":
        # Debug: Print kmeans and region info
        print("kmeans cluster centers:", kmeans.cluster_centers_)
        print("Unique regions in df:", df["region"].unique())
        print("Unique regions in df_plot:", df_plot["region"].unique())

        # Ensure kmeans.transform output is a NumPy array
        distances = kmeans.transform(scaled_cord.to_numpy()).to_numpy().ravel().tolist()
        distances = list(enumerate(distances))
        sorted_distances = sorted(distances, key=lambda x: x[1])[:9]
        indexes = sorted([ind[0] for ind in sorted_distances])

        # Debug: Print indexes
        print("Indexes from kmeans:", indexes)

        # Validate that indexes are present in df["region"]
        valid_regions = df["region"].unique()
        indexes = [ind for ind in indexes if ind in valid_regions]

        if not indexes:
            st.error("No valid regions found for the selected location. Please try a different date/time or check the kmeans model.")
            st.stop()

        df_plot_filtered = df_plot[df_plot["region"].isin(indexes)]

        progress_bar = st.progress(0, text="Operation in progress. Please wait.")
        for percent in range(100):
            sleep(0.05)
            progress_bar.progress(percent + 1, text="Operation in progress. Please wait.")
        st.map(data=df_plot_filtered, latitude="pickup_latitude", longitude="pickup_longitude", size=0.01, color="color")
        progress_bar.empty()

        # Filter input_data safely
        temp_data = df.loc[index, :]
        # Handle case where temp_data is a Series (single row)
        if isinstance(temp_data, pd.Series):
            temp_data = temp_data.to_frame().T
        input_data = temp_data.loc[temp_data["region"].isin(indexes), :].sort_values("region")

        if input_data.empty:
            st.error("No data available for the selected regions at this timestamp. Please try a different date/time.")
            st.stop()

        predictions = pipe.predict(input_data.drop(columns=["total_pickups"]))

        st.markdown("### Map Legend")
        for ind in range(len(indexes)):
            color = colors[indexes[ind]]
            demand = predictions[ind]
            region_id = f"{indexes[ind]} (Current region)" if region == indexes[ind] else indexes[ind]
            st.markdown(f'<div style="display: flex; align-items: center;">'
                        f'<div style="background-color:{color}; width: 20px; height: 10px; margin-right: 10px;"></div>'
                        f'Region ID: {region_id} <br>'
                        f"Demand: {int(demand)} <br> <br>", unsafe_allow_html=True)