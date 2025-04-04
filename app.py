# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors
from io import BytesIO
import threading
import platform
import os
import time

# ==== SOUND FUNCTION ====
def play_sound(path="uploaded-sound.mp3"):
    def _play():
        try:
            from playsound import playsound
            playsound(path)
        except Exception:
            if platform.system() == "Windows":
                os.system(f'start /min wmplayer "{path}"')
            elif platform.system() == "Darwin":
                os.system(f'afplay "{path}"')
            else:
                os.system(f'aplay "{path}"')
    threading.Thread(target=_play, daemon=True).start()

# ==== MOLECULE IMAGE with transparent bg ====
def get_molecule_img_dark(smiles):
    mol = Chem.MolFromSmiles(smiles)
    drawer = Draw.MolDraw2DCairo(300, 300)
    drawer.drawOptions().setBackgroundColour((0, 0, 0, 0))
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()

# ==== SCIENTIFIC NAME ====
def get_scientific_name(smiles):
    known = {
        "CC(=O)OC1=CC=CC=C1C(=O)O": "Acetylsalicylic acid (Aspirin)"
    }
    return known.get(smiles.strip(), "Unknown Compound")

# ==== MOLECULAR PROPERTIES ====
def get_molecular_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return {
        "Molecular Weight": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Descriptors.MolLogP(mol), 2),
        "H-Bond Donors": rdMolDescriptors.CalcNumHBD(mol),
        "H-Bond Acceptors": rdMolDescriptors.CalcNumHBA(mol),
        "Rotatable Bonds": rdMolDescriptors.CalcNumRotatableBonds(mol)
    }

# ==== PAGE CONFIG ====
st.set_page_config(layout="wide", page_title="Quantum AI vs Classical AI")

# ==== LAYOUT ====
left_sidebar, main_area, right_sidebar = st.columns([1, 3, 1], gap="medium")

# === LEFT SIDEBAR: CONTROLS ===
with left_sidebar:
    st.markdown("## ⚙️ Settings")
    if st.button("🔁 Restart Simulation"):
        st.experimental_rerun()

    duration = st.slider("⏱️ Duration (sec)", 5, 30, 10)
    max_candidates = st.slider("🔬 Max Candidates", 100, 10000, 3000)
    smiles = st.text_input("💊 SMILES", "CC(=O)OC1=CC=CC=C1C(=O)O")
    show_molecule = st.checkbox("🧬 Show Molecule on Win", value=True)
    log_scale = st.checkbox("📈 Log Scale (Y-axis)", value=False)

# === MAIN AREA ===
with main_area:
    st.markdown("<h1 style='text-align:center; color:#00FFD5;'>Quantum AI vs Classical AI</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center; color:#999;'>Visualizing Drug Discovery Speed</h4>", unsafe_allow_html=True)

    frames = duration * 5
    times, classical, quantum = [], [], []
    quantum_triggered = False
    graph_area = st.empty()

    start = time.time()

    for i in range(frames):
        elapsed = time.time() - start
        t = i * (duration / frames)
        times.append(t)

        c = int(i * (max_candidates / frames))
        q = min(int(2 ** (i * np.log2(max_candidates) / frames)), max_candidates)
        classical.append(c)
        quantum.append(q)

        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor('#111')
        ax.set_facecolor('#111')
        ax.plot(times, classical, label="Classical AI", color='#FF8C00', linewidth=2.5)
        ax.plot(times, quantum, label="Quantum AI", color='#00FFFF', linewidth=2.5)

        ax.set_xlabel("Time (Seconds)", fontsize=10, color='white')
        ax.set_ylabel("Candidates Processed", fontsize=10, color='white')
        ax.set_title("AI Performance Over Time", color='white', fontsize=14)
        ax.tick_params(axis='both', colors='white', labelsize=9)
        if log_scale:
            ax.set_yscale("log")
        ax.set_ylim(1, max_candidates)
        ax.grid(True, linestyle='--', linewidth=0.5, color='#333')
        ax.spines['top'].set_color('#444')
        ax.spines['right'].set_color('#444')
        ax.spines['left'].set_color('#555')
        ax.spines['bottom'].set_color('#555')
        ax.legend(facecolor='#111', edgecolor='#333', labelcolor='white', fontsize=9)
        graph_area.pyplot(fig)

        if not quantum_triggered and q > c * 3:
            play_sound("uploaded-sound.mp3")
            st.markdown("<h3 style='color:red;'>💥 Quantum AI Breakthrough!</h3>", unsafe_allow_html=True)
            if show_molecule:
                st.image(get_molecule_img_dark(smiles), caption=f"Scientific Name: {get_scientific_name(smiles)}")
            quantum_triggered = True

        time.sleep(0.05)

# === FINAL METRICS ===
q_total = quantum[-1]
c_total = classical[-1]
speedup = round(q_total / c_total, 2)
time_per_molecule_q = round(duration / q_total, 5)
time_per_molecule_c = round(duration / c_total, 5)
mol_props = get_molecular_properties(smiles)
model_scores = {
    "Quantum AI": {"Accuracy": "98.4%", "F1-Score": "0.97", "Energy Est.": "-72.3 kcal/mol"},
    "Classical AI": {"Accuracy": "88.9%", "F1-Score": "0.85", "Energy Est.": "-61.8 kcal/mol"}
}

# === RIGHT SIDEBAR: INFO PANEL ===
with right_sidebar:
    st.markdown("## 📊 AI Performance")
    st.metric("Quantum AI", f"{q_total} molecules", f"{speedup}x faster")
    st.metric("Classical AI", f"{c_total} molecules", "-")
    st.metric("Time per Molecule", f"{time_per_molecule_q}s (Q)", f"{round(time_per_molecule_c / time_per_molecule_q, 1)}x faster")

    st.markdown("## 🧪 Molecular Properties")
    for k, v in mol_props.items():
        st.markdown(f"**{k}:** {v}")

    st.markdown("## 🤖 Model Scores")
    st.markdown("**Quantum AI**")
    for k, v in model_scores["Quantum AI"].items():
        st.markdown(f"- {k}: {v}")

    st.markdown("**Classical AI**")
    for k, v in model_scores["Classical AI"].items():
        st.markdown(f"- {k}: {v}")
