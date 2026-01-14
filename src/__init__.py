"""
SCOPE: Sociopsychological Construct of Persona Evaluation

A framework for constructing and evaluating socially-grounded synthetic personas.

Paper: https://arxiv.org/abs/2601.07110
Authors: Pranav Narayanan Venkit, Yu Li, Yada Pruksachatkun, Chien-Sheng Wu
"""

from .augment_persona import PersonaAugmenter
from .process_nemotron import NemotronProcessor
from .utils import (
    load_jsonl,
    save_jsonl,
    normalize_answer,
    extract_numeric_scale,
    calculate_similarity,
    ResponseParser
)

__version__ = "1.0.0"
__author__ = "Pranav Narayanan Venkit, Yu Li, Yada Pruksachatkun, Chien-Sheng Wu"
__email__ = "pnarayananvenkit@salesforce.com"

__all__ = [
    "PersonaAugmenter",
    "NemotronProcessor",
    "load_jsonl",
    "save_jsonl",
    "normalize_answer",
    "extract_numeric_scale",
    "calculate_similarity",
    "ResponseParser"
]
