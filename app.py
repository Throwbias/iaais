"""Minimal Streamlit smoke test for the IAAIS environment."""

import sys

import streamlit as st
import torch


st.title("Intelligent Adaptive AI System — IAAIS")
st.write(
    "A CSC5350 project exploring adaptive exercise recognition, workout logging, "
    "and intelligent user review."
)
st.write(f"Python version: {sys.version.split()[0]}")
st.write(f"PyTorch version: {torch.__version__}")
st.write(f"CUDA available: {torch.cuda.is_available()}")
st.success("IAAIS environment is operational.")