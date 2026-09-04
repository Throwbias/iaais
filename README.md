# Intelligent Adaptive AI System — IAAIS

IAAIS is a CSC5350 course project exploring an intelligent adaptive AI system. The current concept may use wearable gyroscope and accelerometer data to recognize exercises, estimate repetitions, log workouts, and support user review.

## Planned capabilities

- Symbolic reasoning and knowledge representation
- Search and planning methods
- Machine learning for sensor-based exercise recognition
- Natural language processing for workout interaction and review
- Generative AI features for adaptive assistance
- A Streamlit interface for demonstrations and user review

## WSL setup

From Bash in WSL:

```bash
cd ~/projects/iaais
source iaais_env/bin/activate
```

The project is intentionally independent of the MIPDS project and its environment.

## Dependencies

Install the direct dependencies into the active environment:

```bash
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Jupyter

The environment is registered as the `Python (IAAIS)` kernel. In VS Code, open a notebook, use the kernel picker, and select `Python (IAAIS)`. Run `notebooks/00_environment_check.ipynb` to verify the setup.

## Verification

```bash
python -m pytest
python -m pip check
```

## Streamlit

```bash
streamlit run app.py
```

Never commit `.env`. Keep real API keys out of source control and use `.env.example` only as a template.