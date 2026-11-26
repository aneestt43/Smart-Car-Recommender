# Smart Car Recommender

A Streamlit app that recommends cars based on a user's budget, minimum mileage, seating capacity, and preferred driving terrain. It uses a standardized K-nearest-neighbors (KNN) model to rank cars that meet the selected requirements.

## Features

- Choose a target budget and optionally apply a ₹1 lakh budget buffer.
- Set minimum mileage and seating capacity.
- Select a driving terrain: city, highway, or hills/ghat roads.
- View up to three recommendations and the built-in car catalog.

## How it works

The app filters the catalog by the selected seating capacity and minimum mileage. If the strict budget option is enabled, it also filters out cars priced above the budget plus ₹1 lakh.

It then scales the car features and uses Euclidean distance with KNN to find the closest matches to the user's preferences. Terrain is represented using target ground-clearance and usage values.

## Built with

- Python
- Streamlit
- pandas
- scikit-learn

## Run locally

From Terminal, go to the project folder and install the dependencies:

```bash
cd ~/Desktop/car_app
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
