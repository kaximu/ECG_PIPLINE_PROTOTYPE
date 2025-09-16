# =====================================================
# ECG Diagnosis App (.h5 models, Streamlit)
# =====================================================

import streamlit as st
import numpy as np
import joblib
import json
from pathlib import Path
import tempfile
from scipy.io import loadmat
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pandas as pd

# -----------------------------
# Config
# -----------------------------
RECONSTRUCTOR_H = "ecgnet_reconstructor_best.h5"
CLASSIFIER_H = "ecgnet_with_preprocessing.h5"
CLASS_PKL = "class_names.pkl"
CLASS_JSON = "class_names.json"
DATASET_DIR = Path("datasets")  # fallback for .hea files
TARGET_LEN = 5000  # 10s @ 500 Hz


# =====================================================
# Utilities
# =====================================================
def parse_hea_file(hea_path: Path):
    info = {"Age": "Unknown", "Sex": "Unknown"}
    if not hea_path.exists():
        return info
    for line in hea_path.read_text(errors="ignore").splitlines():
        if not line.startswith("#"):
            continue
        key_val = line[1:].strip()
        if ":" not in key_val:
            continue
        k, v = key_val.split(":", 1)
        if k.strip().lower() == "age":
            info["Age"] = v.strip()
        elif k.strip().lower() in ("sex", "gender"):
            info["Sex"] = v.strip()
    return info


def find_matching_hea(record_id: str, search_dir: Path) -> Path | None:
    candidate = search_dir / f"{record_id}.hea"
    return candidate if candidate.exists() else None


def load_patient_info(mat_path: Path, temp_dir: Path):
    record_id = mat_path.stem
    info = {"Patient ID": record_id, "Age": "Unknown", "Sex": "Unknown"}

    same_folder = find_matching_hea(record_id, temp_dir)
    if same_folder:
        info.update(parse_hea_file(same_folder))
        return info

    dataset_hea = find_matching_hea(record_id, DATASET_DIR)
    if dataset_hea:
        info.update(parse_hea_file(dataset_hea))
        return info

    return info


def preprocess_ecg_3lead(file_path: Path, target_len=TARGET_LEN):
    mat = loadmat(file_path)
    if "val" not in mat:
        raise ValueError("Invalid ECG file: no 'val' key")
    ecg = mat["val"].astype(np.float32)  # (12, N)

    # Normalize per lead
    ecg = (ecg - np.mean(ecg, axis=1, keepdims=True)) / (
        np.std(ecg, axis=1, keepdims=True) + 1e-8
    )

    # Pad or truncate
    n_leads, n_samples = ecg.shape
    if n_samples > target_len:
        ecg = ecg[:, :target_len]
    else:
        pad = np.zeros((n_leads, target_len - n_samples), dtype=np.float32)
        ecg = np.hstack([ecg, pad])

    # Select 3 leads: I, II, V2 → indices 0,1,6
    three_leads = ecg[[0, 1, 6], :]
    return np.expand_dims(three_leads.T, axis=0)  # (1, time, 3)


def plot_ecg(ecg: np.ndarray):
    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF",
                  "V1", "V2", "V3", "V4", "V5", "V6"]
    fig, axes = plt.subplots(12, 1, figsize=(14, 18), sharex=True)
    for i in range(12):
        axes[i].plot(ecg[0, :, i], linewidth=0.8)
        axes[i].set_ylabel(lead_names[i], rotation=0, labelpad=30, fontsize=9)
        axes[i].grid(True, linestyle="--", alpha=0.5)
    axes[-1].set_xlabel("Time (samples)")
    plt.tight_layout()
    return fig


# =====================================================
# Cached Resources (compile=False for inference)
# =====================================================
@st.cache_resource
def load_reconstructor():
    return load_model(RECONSTRUCTOR_H, compile=False)

@st.cache_resource
def load_classifier():
    return load_model(CLASSIFIER_H, compile=False)

@st.cache_resource
def load_classes():
    return joblib.load(CLASS_PKL)

@st.cache_resource
def load_class_fullnames():
    p = Path(CLASS_JSON)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


# =====================================================
# UI
# =====================================================
st.title("🫀 ECG Diagnosis System")
st.write("Upload an ECG `.mat` (and optional `.hea`) → 3 leads [I, II, V2] → reconstructed to 12 → classified.")

st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider("Detection Threshold (%)", 0, 100, 50, 1) / 100.0
top_k = st.sidebar.slider("Top-K predictions", 1, 12, 5, 1)

# Load models/classes
reconstructor = load_reconstructor()
classifier = load_classifier()
class_names = load_classes()
class_fullnames = load_class_fullnames()

# =====================================================
# File Upload
# =====================================================
uploaded_file = st.file_uploader(
    "Upload a .mat file (and optional .hea in the same folder)",
    type=["mat"],
    accept_multiple_files=False
)

if uploaded_file:
    try:
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)

            # ✅ Always overwrite with the new uploaded file
            mat_path = tmpdir / uploaded_file.name
            mat_path.write_bytes(uploaded_file.read())

            # Patient info
            st.subheader("🧑 Patient Information")
            pinfo = load_patient_info(mat_path, tmpdir)
            for k, v in pinfo.items():
                st.write(f"**{k}:** {v}")

            # Pipeline
            X3 = preprocess_ecg_3lead(mat_path)
            ecg_reconstructed = reconstructor.predict(X3)

            preds = classifier.predict(ecg_reconstructed)
            probs = 1 / (1 + np.exp(-preds))  # sigmoid
            probs = probs[0]
            bin_preds = (probs >= threshold).astype(int)

            # Predictions above threshold
            st.subheader("Predictions (above threshold):")
            positive = [class_names[i] for i, v in enumerate(bin_preds) if v == 1]
            if not positive:
                st.warning(f"⚠️ No condition detected above threshold {threshold:.2f}")
            else:
                for i in np.where(bin_preds == 1)[0]:
                    abbr = class_names[i]
                    fullname = class_fullnames.get(abbr, abbr)
                    st.success(f"✅ {abbr} – {fullname} ({probs[i]:.3f})")

            # Top-K
            st.write(f"### Top-{top_k} predictions (by probability)")
            top_idx = np.argsort(probs)[::-1][:top_k]
            results = {class_names[i]: float(probs[i]) for i in top_idx}
            for i in top_idx:
                abbr = class_names[i]
                fullname = class_fullnames.get(abbr, abbr)
                st.write(f"{abbr} – {fullname}: {probs[i]:.3f}")

            # Plot reconstructed ECG
            st.write("### Reconstructed ECG Signal (12 Standard Leads)")
            fig = plot_ecg(ecg_reconstructed)
            st.pyplot(fig)

            # ✅ PDF + CSV download sections here (your updated code)
    except Exception as e:
        st.error(f"Error: {e}")
                # =====================================================
                # Export Buttons
                # =====================================================
            