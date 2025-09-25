import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import joblib, json
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from tensorflow.keras.models import load_model
from supabase import create_client
from typing import Optional, Dict, List

# -----------------------------
# Config
# -----------------------------
RECONSTRUCTOR_H = "ecgnet_reconstructor.h5"
CLASSIFIER_H = "ecgnet_with_preprocessing.h5"
CLASS_PKL = "class_names.pkl"
CLASS_JSON = "class_names.json"
TARGET_LEN = 5000  # fixed length input

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://pbumynpwuptllvjihpia.supabase.co")
SUPABASE_KEY = os.getenv(
    "SUPABASE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBidW15bnB3dXB0bGx2amlocGlhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjYzMDQwNzcsImV4cCI6MjA0MTg4MDA3N30.Ra0j4r_4AtH6U4eZ6JTfascVBmTedusthre-ROg5Lcs",
)  # ⚠️ replace with your service_role key in production
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------------
# Load models/classes
# -----------------------------
@st.cache_resource
def load_reconstructor():
    return load_model(RECONSTRUCTOR_H, compile=False)

@st.cache_resource
def load_classifier():
    return load_model(CLASSIFIER_H, compile=False)

@st.cache_resource
def load_classes() -> List[str]:
    return joblib.load(CLASS_PKL)

@st.cache_resource
def load_class_fullnames() -> Dict[str, str]:
    try:
        with open(CLASS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

reconstructor = load_reconstructor()
classifier = load_classifier()
class_names = load_classes()
class_fullnames = load_class_fullnames()

# -----------------------------
# Utils
# -----------------------------
def preprocess_3lead(lead_i, lead_ii, lead_v2, target_len=TARGET_LEN):
    """Prepare 3-lead ECG as model input (1, time, 3)."""
    try:
        li = np.asarray(lead_i, dtype=np.float32)
        lii = np.asarray(lead_ii, dtype=np.float32)
        lv2 = np.asarray(lead_v2, dtype=np.float32)
    except Exception:
        return None

    n = min(len(li), len(lii), len(lv2))
    if n == 0:
        return None

    x = np.stack([li[:n], lii[:n], lv2[:n]], axis=-1)
    if n > target_len:
        x = x[:target_len]
    else:
        pad = np.zeros((target_len, 3), dtype=np.float32)
        pad[:n] = x
        x = pad
    return np.expand_dims(x, axis=0)

def plot_ecg_12(ecg: np.ndarray):
    """Plot reconstructed 12-lead ECG stacked in 12 rows."""
    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF",
                  "V1", "V2", "V3", "V4", "V5", "V6"]
    fig, axes = plt.subplots(12, 1, figsize=(12, 18), sharex=True)
    for i in range(12):
        axes[i].plot(ecg[0, :, i], linewidth=0.8)
        axes[i].set_ylabel(lead_names[i], rotation=0, labelpad=30, fontsize=8)
        axes[i].grid(True, linestyle="--", alpha=0.5)
    axes[-1].set_xlabel("Time (samples)")
    plt.tight_layout()
    return fig

# -----------------------------
# UI
# -----------------------------
st.title("🫀 ECG Diagnosis System (Supabase Data)")

st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider("Detection Threshold (%)", 0, 100, 50, 1) / 100.0
top_k = st.sidebar.slider("Top-K predictions", 1, 12, 5, 1)

# Load patients + records
patients = supabase.table("patients").select("*").execute().data
records = supabase.table("ecg_records").select("*").execute().data
df_patients = pd.DataFrame(patients)
df_records = pd.DataFrame(records)

if df_patients.empty or df_records.empty:
    st.warning("⚠️ No patients or ECG data found in Supabase.")
    st.stop()

# Select patient
selected_patient = st.sidebar.selectbox("Select patient", df_patients["name"].tolist())
patient_info = df_patients[df_patients["name"] == selected_patient].iloc[0]

st.subheader("👤 Patient Information")
st.write(f"**Name:** {patient_info['name']}")
st.write(f"**Age:** {patient_info['age']}")
st.write(f"**Gender:** {patient_info['gender']}")

# Filter ECGs
patient_ecg = df_records[df_records["patient_id"] == patient_info["id"]]
if patient_ecg.empty:
    st.info("ℹ️ No ECGs linked to this patient.")
    st.stop()

# Select ECG record
record_options = [f"{i+1} | {row['created_at'][:19]} | {row['id']}"
                  for i, (_, row) in enumerate(patient_ecg.iterrows())]
selected = st.selectbox("Select ECG record", record_options)
sel_idx = record_options.index(selected)
record = list(patient_ecg.iterrows())[sel_idx][1]

if not all(k in record for k in ["lead_i", "lead_ii", "lead_v2"]):
    st.error("❌ This record does not have 3-lead data.")
    st.stop()

# -----------------------------
# Pipeline
# -----------------------------
X3 = preprocess_3lead(record["lead_i"], record["lead_ii"], record["lead_v2"])
if X3 is None:
    st.error("❌ Could not preprocess ECG leads.")
    st.stop()

try:
    ecg_reconstructed = reconstructor.predict(X3)
except Exception as e:
    st.error(f"❌ Reconstruction failed: {e}")
    st.stop()

try:
    preds = classifier.predict(ecg_reconstructed)
    probs = 1 / (1 + np.exp(-preds))  # sigmoid
    probs = probs[0]
except Exception as e:
    st.error(f"❌ Classification failed: {e}")
    st.stop()

if len(probs) == 0:
    st.error("❌ No predictions computed.")
    st.stop()

bin_preds = (probs >= threshold).astype(int)

# Predictions
st.subheader("🩺 Predictions")
positive = [class_names[i] for i, v in enumerate(bin_preds) if v == 1]
if not positive:
    st.info(f"⚠️ No condition detected above threshold {threshold:.2f} → defaulting to **Normal**")
    pred_text = "Normal"
    st.success("✅ Normal – Normal ECG (0.500)")
else:
    pred_text = ", ".join(positive)
    for i in np.where(bin_preds == 1)[0]:
        abbr = class_names[i]
        fullname = class_fullnames.get(abbr, abbr)
        st.success(f"✅ {abbr} – {fullname} ({probs[i]:.3f})")

# Top-K
st.write(f"### Top-{top_k} predictions")
top_idx = np.argsort(probs)[::-1][:top_k]
for i in top_idx:
    abbr = class_names[i]
    fullname = class_fullnames.get(abbr, abbr)
    st.write(f"{abbr} – {fullname}: {probs[i]:.3f}")

# -----------------------------
# ECG Plot
# -----------------------------
st.write("### Reconstructed ECG (12 Standard Leads)")
fig = plot_ecg_12(ecg_reconstructed)
st.pyplot(fig)

# -----------------------------
# PDF Export (on demand)
# -----------------------------
if st.button("📄 Download ECG Report as PDF"):
    pdf_buffer = io.BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        # Header page
        fig_header = plt.figure(figsize=(8.5, 5))
        plt.axis("off")
        plt.text(0.5, 0.9, "ECG DIAGNOSIS REPORT", fontsize=20,
                 fontweight="bold", ha="center", color="darkblue")
        y = 0.75
        info_lines = [
            ("System", "ECG Diagnosis System"),
            ("Report Date", datetime.now().strftime("%Y-%m-%d %H:%M")),
            ("Patient ID", patient_info.get("id", "Unknown")),
            ("Name", patient_info.get("name", "Unknown")),
            ("Age", patient_info.get("age", "Unknown")),
            ("Gender", patient_info.get("gender", "Unknown")),
            ("Prediction", pred_text),
        ]
        for label, val in info_lines:
            plt.text(0.1, y, f"{label}:", fontsize=12, fontweight="bold")
            plt.text(0.35, y, str(val), fontsize=12)
            y -= 0.08

        # Top-K predictions
        y -= 0.05
        plt.text(0.1, y, f"Top-{top_k} predictions:", fontsize=12,
                 fontweight="bold", color="green")
        y -= 0.06
        for idx in top_idx[:top_k]:
            abbr = class_names[idx]
            fullname = class_fullnames.get(abbr, abbr)
            prob = probs[idx]
            plt.text(0.12, y, f"{abbr} – {fullname}: {prob:.3f}", fontsize=11)
            y -= 0.05

        pdf.savefig(fig_header)
        plt.close(fig_header)

        # ECG plot page
        pdf.savefig(fig)

    pdf_buffer.seek(0)
    st.download_button(
        label="⬇️ Save Report as PDF",
        data=pdf_buffer,
        file_name="ecg_report.pdf",
        mime="application/pdf"
    )
