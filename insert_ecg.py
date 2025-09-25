from supabase import create_client
import numpy as np

# ----------------------------
# 1. Connect to Supabase
# ----------------------------
url = "https://pbumynpwuptllvjihpia.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBidW15bnB3dXB0bGx2amlocGlhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjYzMDQwNzcsImV4cCI6MjA0MTg4MDA3N30.Ra0j4r_4AtH6U4eZ6JTfascVBmTedusthre-ROg5Lcs"   # or anon key for frontend
supabase = create_client(url, key)

# ----------------------------
# 2. Insert a patient
# ----------------------------
patient = supabase.table("patients").insert({
    "name": "John Doe",
    "age": 54,
    "gender": "male"
}).execute()

patient_id = patient.data[0]["id"]
print("✅ New patient ID:", patient_id)

# ----------------------------
# 3. Insert an ECG record
# ----------------------------
# Simulated ECG signal (500 samples)
ecg_signal = np.random.randn(500).tolist()

ecg_record = supabase.table("ecg_records").insert({
    "patient_id": patient_id,
    "sampling_rate": 500,
    "signal": ecg_signal,
    "prediction": "Normal sinus rhythm"
}).execute()

print("✅ New ECG record:", ecg_record.data[0]["id"])
