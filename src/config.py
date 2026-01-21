"""Configuration constants for the Fake News Detector app."""

import numpy as np

PAGE_CONFIG = {
    "page_title": "Fake News Detector",
    "page_icon": "🔍",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

SIDEBAR_CSS = """
<style>
    [data-testid = "stSidebar"] {
        min-width: 380px;
        max-width: 420px;
    }
</style>
"""

MODEL_METRICS = {
    'Accuracy': 0.9803,
    'Precision': 0.9850,
    'Recall': 0.9766,
    'F1-Score': 0.9808,
    'ROC-AUC': 0.9979
}

DATASET_DISTRIBUTION = {
    'sizes': [48.7, 51.3],
    'labels': ['FAKE', 'REAL']
}

CONFUSION_MATRIX = np.array([[9595, 196], [243, 10096]])

COLORS = {
    'metrics': ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12'],
    'pie': ['#e74c3c', '#2ecc71']
}
