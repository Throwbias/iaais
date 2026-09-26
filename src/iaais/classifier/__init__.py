"""Supervised prediction for the IAAIS exercise-recognition domain."""

from .classifier import Classifier
from .features import (
    InertialFeatureEngineer,
    KnowledgeBaseFeatureExtractor,
    write_sensor_features_to_knowledge_base,
)
from .models import (
    ClassifierEvaluation,
    ClassifierPrediction,
    FeatureContribution,
    FeatureFactSpec,
    FeatureVector,
    TrainingExample,
)

__all__ = [
    "Classifier",
    "ClassifierEvaluation",
    "ClassifierPrediction",
    "FeatureContribution",
    "FeatureFactSpec",
    "FeatureVector",
    "InertialFeatureEngineer",
    "KnowledgeBaseFeatureExtractor",
    "TrainingExample",
    "write_sensor_features_to_knowledge_base",
]
