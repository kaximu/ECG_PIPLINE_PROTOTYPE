from supabase import create_client
import numpy as np

# ----------------------------
# 1. Connect to Supabase
# ----------------------------
url = "https://<YOUR_PROJECT>.supabase.co"
key = "<YOUR_SERVICE_ROLE_KEY>"   # or anon key for frontend
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
