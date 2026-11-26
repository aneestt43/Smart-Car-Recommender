import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


st.set_page_config(page_title="AI Car Recommender", page_icon="🚗", layout="wide")
st.title("🚗 Smart Car Recommender")
st.markdown(
    "Adjust your budget and driving needs to get recommendations ranked with a "
    "standardized K-nearest-neighbors model."
)

car_data = {
    "Car_Name": [
        "Maruti Alto K10", "Hyundai Grand i10 Nios", "Tata Tiago EV",
        "Maruti Swift", "Tata Punch", "Hyundai i20", "Maruti Brezza",
        "Mahindra XUV 3XO", "Hyundai Creta", "Toyota Innova Crysta",
        "Mahindra Thar Roxx", "Tata Nexon EV",
    ],
    "Price_Lakhs": [5.5, 7.5, 9.0, 8.0, 8.5, 9.5, 11.5, 12.0, 15.0, 22.0, 18.0, 14.5],
    "Mileage_kmpl": [24.0, 20.0, 35.0, 22.0, 19.0, 18.0, 19.0, 18.5, 17.0, 14.0, 15.0, 32.0],
    "Seats": [5, 5, 5, 5, 5, 5, 5, 5, 5, 7, 5, 5],
    "Ground_Clearance_mm": [165, 165, 165, 163, 187, 170, 198, 201, 190, 178, 226, 190],
    "Usage_Type": [1, 1, 1, 1, 3, 2, 2, 3, 2, 2, 3, 1],
    "Fuel_Type": [
        "Petrol", "Petrol", "Electric", "Petrol", "Petrol", "Petrol",
        "Petrol", "Petrol/Diesel", "Petrol/Diesel", "Diesel",
        "Diesel/Petrol", "Electric",
    ],
}
df = pd.DataFrame(car_data)

terrain_options = {
    "City Commute": (1, 165),
    "Highway Cruising": (2, 180),
    "Hills / Ghat Roads": (3, 195),
}

st.sidebar.header("⚙️ Your Requirements")
budget = st.sidebar.slider(
    "Target Budget (₹ Lakhs)", min_value=5.0, max_value=25.0, value=9.0, step=0.5
)
strict_budget = st.sidebar.checkbox(
    "Strictly under my budget (+ ₹1 Lakh buffer)", value=False
)
use_case = st.sidebar.selectbox("Primary Driving Terrain", list(terrain_options))
mileage = st.sidebar.slider(
    "Minimum Mileage / Efficiency (kmpl)", min_value=12, max_value=35, value=18
)
seats = st.sidebar.radio("Seating Capacity", options=[5, 7], horizontal=True)

filtered_df = df.copy()
if strict_budget:
    filtered_df = filtered_df[filtered_df["Price_Lakhs"] <= budget + 1.0]
filtered_df = filtered_df[
    (filtered_df["Mileage_kmpl"] >= mileage) & (filtered_df["Seats"] == seats)
].copy()

st.subheader("🏆 Top 3 Recommendations")
if filtered_df.empty:
    st.info(
        "No cars meet all of those requirements. Try lowering the minimum mileage, "
        "changing the seating capacity, or relaxing the strict budget filter."
    )
else:
    usage_code, target_clearance = terrain_options[use_case]
    feature_columns = [
        "Price_Lakhs", "Mileage_kmpl", "Seats", "Ground_Clearance_mm", "Usage_Type"
    ]
    target = pd.DataFrame(
        [[budget, mileage, seats, target_clearance, usage_code]],
        columns=feature_columns,
    )

    # Scale candidates and the user's request together so each feature contributes
    # on a comparable scale to the Euclidean distance.
    scaler = StandardScaler()
    candidate_features = filtered_df[feature_columns]
    combined = pd.concat([candidate_features, target], ignore_index=True)
    scaled = scaler.fit_transform(combined)
    candidate_scaled, target_scaled = scaled[:-1], scaled[-1:]

    neighbors = min(3, len(filtered_df))
    model = NearestNeighbors(n_neighbors=neighbors, metric="euclidean")
    model.fit(candidate_scaled)
    distances, indices = model.kneighbors(target_scaled)
    recommendations = filtered_df.iloc[indices[0]].copy()
    recommendations["Distance"] = distances[0]

    cards = st.columns(3)
    for rank, (_, car) in enumerate(recommendations.iterrows(), start=1):
        with cards[rank - 1]:
            st.markdown(f"### {rank}. {car['Car_Name']}")
            st.metric("Price", f"₹{car['Price_Lakhs']:.1f} lakh")
            st.write(f"**Fuel:** {car['Fuel_Type']}")
            st.write(f"**Efficiency:** {car['Mileage_kmpl']:.1f} kmpl")
            st.write(f"**Ground clearance:** {car['Ground_Clearance_mm']} mm")
            st.write(f"**Seats:** {car['Seats']}")

with st.expander("View the built-in car catalog"):
    st.dataframe(df, hide_index=True, use_container_width=True)
