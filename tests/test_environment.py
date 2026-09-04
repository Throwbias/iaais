"""Lightweight checks for the IAAIS development environment."""

import importlib

import spacy


def test_core_libraries_import():
    libraries = [
        "jupyter",
        "ipykernel",
        "sklearn",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "networkx",
        "spacy",
        "nltk",
        "transformers",
        "torch",
        "openai",
        "streamlit",
        "dotenv",
    ]
    for library in libraries:
        assert importlib.import_module(library) is not None


def test_spacy_model_loads():
    nlp = spacy.load("en_core_web_sm")
    assert nlp("IAAIS is ready.").text == "IAAIS is ready."