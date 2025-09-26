import streamlit as st
import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
import time
import random


# -----------------------------
# ECG Simulation Function
# -----------------------------
def simulate_ecg(condition="normal", duration=10, fs=500):
    if condition == "normal":
        ecg = nk.ecg_simulate(duration=duration, sampling_rate=fs, heart_rate=75)

    elif condition == "bradycardia":
        ecg = nk.ecg_simulate(duration=duration, sampling_rate=fs, heart_rate=40)

    elif condition == "tachycardia":
        ecg = nk.ecg_simulate(duration=duration, sampling_rate=fs, heart_rate=150)

    elif condition == "afib":
        hr_series = np.random.randint(60, 150, size=duration)  # random HR each second
        ecg = np.concatenate([
            nk.ecg_simulate(duration=1, sampling_rate=fs, heart_rate=hr) for hr in hr_series
        ])

    elif condition == "pvc":
        ecg = nk.ecg_simulate(duration=duration, sampling_rate=fs, heart_rate=75)
        for _ in range(duration // 3):  # insert PVC beats
            pos = random.randint(fs, len(ecg) - fs)
            pvc_shape = -1.5 * np.exp(-((np.arange(fs) - fs//2)**2) / (2*(fs//20)**2))
            ecg[pos:pos+fs] = ecg[pos:pos+fs] * 0.5 + pvc_shape[:len(ecg[pos:pos+fs])]

    else:
        raise ValueError("Condition not recognized.")

    return ecg


def stream_ecg(ecg, fs=500, chunk_sec=1):
    chunk_size = fs * chunk_sec
    for i in range(0, len(ecg), chunk_size):
        chunk = ecg[i:i+chunk_size]
        yield chunk
        time.sleep(chunk_sec)


# -----------------------------
# Streamlit App
# -----------------------------
st.title("🩺 ECG Simulator")

# Select condition
condition = st.selectbox(
    "Choose condition to simulate:",
    ["normal", "bradycardia", "tachycardia", "afib", "pvc"]
)

# Parameters
duration = st.slider("Duration (seconds)", 5, 60, 10, step=5)
fs = st.number_input("Sampling Rate (Hz)", value=500, step=50)

if st.button("Generate ECG"):
    st.write(f"Simulating **{condition.upper()}** for {duration} sec at {fs} Hz...")
    
    ecg = simulate_ecg(condition=condition, duration=duration, fs=fs)

    # Plot ECG (first 5 seconds if long)
    plot_len = min(len(ecg), fs*5)
    st.line_chart(ecg[:plot_len])

    # Option to stream
    if st.checkbox("Stream ECG in real-time"):
        placeholder = st.empty()
        for chunk in stream_ecg(ecg, fs=fs, chunk_sec=1):
            placeholder.line_chart(chunk)
