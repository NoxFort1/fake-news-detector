"""Model loading and prediction functions."""

import pickle
import numpy as np
import streamlit as st
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences

from text_features import preprocess_text, TextFeatureExtractor


_feature_extractor = None


def get_feature_extractor() -> TextFeatureExtractor:
    """Get or create the feature extractor singleton."""
    global _feature_extractor
    if _feature_extractor is None:
        _feature_extractor = TextFeatureExtractor()
    return _feature_extractor


@st.cache_resource
def load_model():
    """
    Load the trained model, tokenizer, and configuration.
    
    Returns:
        tuple: (model, tokenizer, config) or (None, None, None) on failure
    """
    try:
        model = keras.models.load_model('models/cnn_lstm_mlp_model.keras')

        with open('models/tokenizer.pkl', 'rb') as file:
            tokenizer = pickle.load(file)

        with open('models/config.pkl', 'rb') as file:
            config = pickle.load(file)

        return model, tokenizer, config
    except Exception as e:
        st.error(f'Error loading model: {e}')
        return None, None, None


def build_meta_features(text: str, feature_columns: list) -> np.ndarray:
    """
    Extract metadata features from text.
    
    Args:
        text: The input text
        feature_columns: List of feature column names
        
    Returns:
        numpy array of features
    """
    feature_extractor = get_feature_extractor()
    features = feature_extractor.extract_features(text)
    return np.array([features.get(name, 0.0) for name in feature_columns], dtype = np.float32)


def predict_tweet(text: str, model, tokenizer, config) -> dict | None:
    """
    Predict whether a tweet is fake or real.
    
    Args:
        text: The tweet text to analyze
        model: The loaded Keras model
        tokenizer: The loaded tokenizer
        config: The model configuration dictionary
        
    Returns:
        Dictionary with prediction results or None on failure
    """
    if model is None or tokenizer is None or config is None:
        return None

    try:
        cleaned_text = preprocess_text(text)
        sequence = tokenizer.texts_to_sequences([cleaned_text])

        vocab_size = config.get('vocab_size', 10000)
        sequence = [[token if token <= vocab_size else 0 for token in seq] for seq in sequence]

        padded_text = pad_sequences(
            sequence,
            maxlen = config['max_len'],
            padding = 'post',
            truncating = 'post'
        )

        feature_columns = config.get('feature_columns', [])
        meta = build_meta_features(text, feature_columns)
        meta = np.expand_dims(meta, axis = 0)

        probability = model.predict([padded_text, meta], verbose = 0)[0][0]
        is_real = probability >= 0.5
        confidence = probability if is_real else (1 - probability)

        return {
            'verdict': 'REAL' if is_real else 'FAKE',
            'confidence': float(confidence * 100),
            'real_prob': float(probability * 100),
            'fake_prob': float((1 - probability) * 100),
            'is_real': bool(is_real)
        }
    except Exception as e:
        st.error(f'Error during prediction: {e}')
        return None
