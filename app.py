import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# 1. Load Data & Train Model (Simple Pipeline)
@st.cache_resource
def load_and_train():
    # Generate/load the underlying dataset
    # In a production app, you would load the CSV file using pd.read_csv()
    try:
        df = pd.read_csv('delivery_delay.csv')
    except FileNotFoundError:
        # Backup dummy generation in case CSV path is different on deployment
        st.warning("delivery_delay.csv not found, using dummy training dataset.")
        np.random.seed(42)
        data = {
            'Delivery_Distance': np.random.uniform(5, 50, 1000),
            'Traffic_Congestion': np.random.randint(1, 6, 1000),
            'Weather_Condition': np.random.randint(1, 4, 1000),
            'Delivery_Slot': np.random.randint(1, 4, 1000),
            'Driver_Experience': np.random.randint(1, 20, 1000),
            'Num_Stops': np.random.randint(1, 10, 1000),
            'Vehicle_Age': np.random.randint(1, 10, 1000),
            'Road_Condition_Score': np.random.randint(1, 5, 1000),
            'Package_Weight': np.random.uniform(1, 40, 1000),
            'Fuel_Efficiency': np.random.uniform(5, 20, 1000),
            'Warehouse_Processing_Time': np.random.randint(10, 90, 1000),
            'Delivery_Delay': np.random.randint(0, 2, 1000)
        }
        df = pd.DataFrame(data)

    X = df.drop(columns=['Delivery_Delay'])
    y = df['Delivery_Delay'].astype(str)
    
    clf = DecisionTreeClassifier(random_state=42, max_depth=5)
    clf.fit(X, y)
    return clf, X.columns.tolist()

clf, feature_cols = load_and_train()

# 2. Build Streamlit UI
st.title("🚚 Delivery Delay Prediction App")
st.write("Enter the delivery parameters below to predict if a delivery will be delayed.")

# Input Forms
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        distance = st.number_input("Delivery Distance (km)", min_value=0.0, value=25.0)
        traffic = st.slider("Traffic Congestion Level", min_value=1, max_value=5, value=3)
        weather = st.slider("Weather Condition Level", min_value=1, max_value=3, value=2)
        slot = st.slider("Delivery Slot", min_value=1, max_value=3, value=2)
        experience = st.number_input("Driver Experience (Years)", min_value=0, value=10)
        stops = st.number_input("Number of Stops", min_value=0, value=3)
        
    with col2:
        vehicle_age = st.number_input("Vehicle Age (Years)", min_value=0, value=5)
        road_score = st.slider("Road Condition Score", min_value=1, max_value=4, value=3)
        weight = st.number_input("Package Weight (kg)", min_value=0.0, value=20.0)
        fuel = st.number_input("Fuel Efficiency (km/l)", min_value=0.0, value=12.0)
        warehouse_time = st.number_input("Warehouse Processing Time (mins)", min_value=0, value=45)
        
    submit = st.form_submit_button("Predict Delay")

# 3. Predict and Display Results
if submit:
    input_data = pd.DataFrame([[
        distance, traffic, weather, slot, experience, stops, 
        vehicle_age, road_score, weight, fuel, warehouse_time
    ]], columns=feature_cols)
    
    prediction = clf.predict(input_data)[0]
    probability = clf.predict_proba(input_data)
    
    if str(prediction) == '1':
        st.error(f"🚨 Prediction: Delayed (Probability: {probability[0][1]*100:.1f}%)")
    else:
        st.success(f"✅ Prediction: No Delay (Probability: {probability[0][0]*100:.1f}%)")

with open("app.py", "w") as f:
    f.write(app_code)

print("Streamlit app successfully saved to app.py")
