import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from supabase import create_client

# ----------------------------
# 1. Connect to Supabase
# ----------------------------
url = "https://<YOUR_PROJECT>.supabase.co"   # 🔑 replace with your project URL
key = "<YOUR_SERVICE_ROLE_KEY>"              # 🔑 replace with your API key (service_role for backend)
supabase = create_client(url, key)

# ----------------------------
# 2. Fetch Data
# ----------------------------
def load_data():
    patients = supabase.table("patients").select("*").execute().data
    ecg_records = supabase.table("ecg_records").select("*").execute().data
    return pd.DataFrame(patients), pd.DataFrame(ecg_records)

df_patients, df_ecg = load_data()

# ----------------------------
# 3. Streamlit Dashboard
# ----------------------------
st.set_page_config(page_title="ECG Dashboard", layout="wide")
st.title("🫀 Patient ECG Dashboard")

if df_patients.empty:
    st.warning("⚠️ No patients found in database. Insert some data first.")
else:
    # Sidebar: choose patient
    patient_list = df_patients["name"].tolist()
    selected_patient = st.sidebar.selectbox("Select a patient:", patient_list)

    # Get patient info
    patient_info = df_patients[df_patients["name"] == selected_patient].iloc[0]
    st.subheader(f"👤 Patient Information")
    st.write(f"- **Name:** {patient_info['name']}")
    st.write(f"- **Age:** {patient_info['age']}")
    st.write(f"- **Gender:** {patient_info['gender']}")

    # Filter ECG data for this patient
    patient_ecg = df_ecg[df_ecg["patient_id"] == patient_info["id"]]

    if patient_ecg.empty:
        st.warning("⚠️ No ECG data available for this patient.")
    else:
        st.subheader("📊 ECG Records")

        # Show table of records
        st.dataframe(patient_ecg[["created_at", "sampling_rate", "prediction"]])

        # Plot signals
        for _, record in patient_ecg.iterrows():
            try:
                signal = np.array(record["signal"])
                fig = px.line(
                    y=signal,
                    title=f"ECG Signal | Prediction: {record['prediction']} | Sampling Rate: {record['sampling_rate']} Hz"
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error plotting ECG record {record['id']}: {e}")
