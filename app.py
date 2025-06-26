import joblib
import streamlit as st
import pandas as pd
import numpy as np
import json

st.write("🚀 App started loading...")

# Load the model and feature list

model = joblib.load("model.joblib")
with open('model.pickle', 'rb') as model_pickle:
        model = pickle.load(model_pickle)
    
with open("features.json", "r") as g:
    features = json.load(g)

st.title("📱 Expresso Churn Prediction App")

# Input fields
st.header("Enter Customer Data")

region_input = st.selectbox("REGION", [
    'DAKAR', 'THIES', 'SAINT-LOUIS', 'DIOURBEL', 'LOUGA', 'FATICK',
    'KAOLACK', 'ZIGUINCHOR', 'KOLDA', 'TAMBACOUNDA', 'MATAM', 'SEDHIOU', 'KEDOUGOU'
])

tenure_input = st.selectbox("TENURE", [
    'K > 24 month', 'J 21-24 month', 'I 18-21 month', 'H 15-18 month',
    'G 12-15 month', 'F 9-12 month', 'E 6-9 month', 'D 3-6 month'
])

montant = st.number_input("MONTANT", min_value=0.0, value=1000.0)
frequence_rech = st.number_input("FREQUENCE_RECH", min_value=0, value=5)
revenue = st.number_input("REVENUE", min_value=0.0, value=1000.0)
arpu_segment = st.number_input("ARPU_SEGMENT", min_value=0.0, value=1000.0)
frequence = st.number_input("FREQUENCE", min_value=0, value=5)
data_volume = st.number_input("DATA_VOLUME", min_value=0.0, value=500.0)
on_net = st.number_input("ON_NET", min_value=0.0, value=300.0)
orange = st.number_input("ORANGE", min_value=0.0, value=200.0)
tigo = st.number_input("TIGO", min_value=0.0, value=100.0)
zone1 = st.number_input("ZONE1", min_value=0.0, value=50.0)
zone2 = st.number_input("ZONE2", min_value=0.0, value=50.0)
mrg = st.selectbox("MRG", ['NO'])
regularity = st.number_input("REGULARITY", min_value=0, value=5)
top_pack_input = st.selectbox("TOP_PACK", [
    'All-net 1000F=5000F;7d', 'Sama Pass 500F=25Min', 'All-net 200F=30Min',
    'All-net 2000F=10000F;7d'
])
freq_top_pack = st.number_input("FREQ_TOP_PACK", min_value=0, value=1)

# Preprocessing (replicating encoding done during training)
region_map = {'DAKAR': 0, 'THIES': 1, 'SAINT-LOUIS': 2, 'DIOURBEL': 3, 'LOUGA': 4, 'FATICK': 5,
              'KAOLACK': 6, 'ZIGUINCHOR': 7, 'KOLDA': 8, 'TAMBACOUNDA': 9, 'MATAM': 10,
              'SEDHIOU': 11, 'KEDOUGOU': 12}
tenure_map = {'K > 24 month': 474264, 'J 21-24 month': 25000, 'I 18-21 month': 20000,
              'H 15-18 month': 18000, 'G 12-15 month': 16000, 'F 9-12 month': 14000,
              'E 6-9 month': 12000, 'D 3-6 month': 10000}
top_pack_map = {
    'All-net 1000F=5000F;7d': 5867, 'Sama Pass 500F=25Min': 4000,
    'All-net 200F=30Min': 3000, 'All-net 2000F=10000F;7d': 2500
}

# Map values
region = region_map.get(region_input, -1)
tenure = tenure_map.get(tenure_input, 10000)
top_pack = top_pack_map.get(top_pack_input, 1000)
mrg = 400000 if mrg == 'NO' else 0
tf = regularity * frequence
mr = montant - revenue

# Start feature dict
input_dict = {
    'REGION': region,
    'TENURE': tenure,
    'MONTANT': montant,
    'FREQUENCE_RECH': frequence_rech,
    'REVENUE': revenue,
    'ARPU_SEGMENT': arpu_segment,
    'FREQUENCE': frequence,
    'DATA_VOLUME': data_volume,
    'ON_NET': on_net,
    'ORANGE': orange,
    'TIGO': tigo,
    'ZONE1': zone1,
    'ZONE2': zone2,
    'MRG': mrg,
    'REGULARITY': regularity,
    'TOP_PACK': top_pack,
    'FREQ_TOP_PACK': freq_top_pack,
    'TF': tf,
    'MR': mr
}

# Ensure all features used in training are present
for col in features:
    if col not in input_dict:
        input_dict[col] = 0

# Make sure column order matches training
X = pd.DataFrame([input_dict])[features]

# Predict
if st.button("Predict Churn Probability"):
    churn_prob = model.predict_proba(X)[0][1]
    st.success(f"Predicted Churn Probability: {churn_prob:.4f}")
    if churn_prob > 0.5:
        st.warning("⚠️ High risk of churn!")
    else:
        st.info("✅ Low risk of churn.")
