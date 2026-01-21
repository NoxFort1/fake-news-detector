"""
Text Feature Extraction Module for Fake News Detection

This module extracts linguistic and statistical features from text
for use with the fake news detection model.
"""

import re
import spacy
from collections import Counter
from typing import Dict, List, Optional
import numpy as np


class TextFeatureExtractor:
    """
    Extracts metadata features from text for fake news detection.
    
    Features extracted:
    - Linguistic features (POS tags, NER entities)
    - Statistical features (word count, character counts)
    - Social media specific features (hashtags, URLs, mentions)
    """
    
    FEATURE_ORDER = [
        'statuses_count', 'following', 'cred', 'normalize_influence',
        'hashtags', 'urls', 'unique_count', 'NORP_percentage',
        'PERSON_percentage', 'MONEY_percentage', 'CARDINAL_percentage',
        'word_count', 'present_verbs', 'past_verbs', 'adjectives',
        'pronouns', 'tos', 'exclamation', 'capitals', 'short_word_freq'
    ]
    
    DEFAULT_USER_FEATURES = {
        'statuses_count': 1000.0,
        'following': 1.0,
        'cred': 0.5,
        'normalize_influence': 0.5
    }
    
    def __init__(self, spacy_model: str = 'en_core_web_sm'):
        """
        Initialize the feature extractor.
        
        Args:
            spacy_model: Name of the spaCy model to use for NLP tasks.
        """
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"Downloading spaCy model '{spacy_model}'...")
            spacy.cli.download(spacy_model)
            self.nlp = spacy.load(spacy_model)
    
    def extract_features(
        self, 
        text: str, 
        user_features: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Extract all features from the given text.
        
        Args:
            text: The input text to analyze.
            user_features: Optional dict with user profile features
                          (statuses_count, following, cred, normalize_influence).
        
        Returns:
            Dictionary containing all extracted features.
        """
        doc = self.nlp(text)
        words = [token.text for token in doc if not token.is_space]
        
        features = {}
        
        user_feats = user_features or {}
        for key, default_val in self.DEFAULT_USER_FEATURES.items():
            features[key] = user_feats.get(key, default_val)
        
        features['hashtags'] = self._count_hashtags(text)
        features['urls'] = self._count_urls(text)
        
        features['unique_count'] = len(set(word.lower() for word in words))
        features['word_count'] = len(words)
        
        ner_percentages = self._extract_ner_percentages(doc)
        features['NORP_percentage'] = ner_percentages.get('NORP', 0.0)
        features['PERSON_percentage'] = ner_percentages.get('PERSON', 0.0)
        features['MONEY_percentage'] = ner_percentages.get('MONEY', 0.0)
        features['CARDINAL_percentage'] = ner_percentages.get('CARDINAL', 0.0)
        
        pos_counts = self._extract_pos_counts(doc)
        features['present_verbs'] = pos_counts.get('present_verbs', 0)
        features['past_verbs'] = pos_counts.get('past_verbs', 0)
        features['adjectives'] = pos_counts.get('adjectives', 0)
        features['pronouns'] = pos_counts.get('pronouns', 0)
        features['tos'] = pos_counts.get('tos', 0)
        
        features['exclamation'] = text.count('!')
        features['capitals'] = self._count_capitals(text)
        features['short_word_freq'] = self._count_short_words(words)
        
        return features
    
    def extract_features_array(
        self, 
        text: str, 
        user_features: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Extract features and return as numpy array in the correct order.
        
        Args:
            text: The input text to analyze.
            user_features: Optional dict with user profile features.
        
        Returns:
            Numpy array of features in the expected order for the model.
        """
        features = self.extract_features(text, user_features)
        return np.array([features[key] for key in self.FEATURE_ORDER], dtype=np.float32)
    
    def _count_hashtags(self, text: str) -> int:
        """Count the number of hashtags in the text."""
        return len(re.findall(r'#\w+', text))
    
    def _count_urls(self, text: str) -> int:
        """Count the number of URLs in the text."""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return len(re.findall(url_pattern, text))
    
    def _extract_ner_percentages(self, doc) -> Dict[str, float]:
        """
        Extract Named Entity Recognition percentages.
        
        Returns percentage of text tokens that belong to each entity type.
        """
        if len(doc) == 0:
            return {}
        
        entity_counts = Counter()
        total_entity_tokens = 0
        
        for ent in doc.ents:
            entity_counts[ent.label_] += len(ent)
            total_entity_tokens += len(ent)
        
        total_tokens = len(doc)
        percentages = {}
        for entity_type, count in entity_counts.items():
            percentages[entity_type] = (count / total_tokens) * 100 if total_tokens > 0 else 0.0
        
        return percentages
    
    def _extract_pos_counts(self, doc) -> Dict[str, int]:
        """
        Extract Part-of-Speech tag counts.
        
        Returns:
            Dictionary with counts of different POS categories.
        """
        counts = {
            'present_verbs': 0,
            'past_verbs': 0,
            'adjectives': 0,
            'pronouns': 0,
            'tos': 0
        }
        
        for token in doc:
            # Present verbs (VBP, VBZ, VBG)
            if token.tag_ in ['VBP', 'VBZ', 'VBG']:
                counts['present_verbs'] += 1
            # Past verbs (VBD, VBN)
            elif token.tag_ in ['VBD', 'VBN']:
                counts['past_verbs'] += 1
            # Adjectives (JJ, JJR, JJS)
            elif token.tag_ in ['JJ', 'JJR', 'JJS']:
                counts['adjectives'] += 1
            # Pronouns (PRP, PRP$, WP, WP$)
            elif token.tag_ in ['PRP', 'PRP$', 'WP', 'WP$']:
                counts['pronouns'] += 1
            # "TO" particle
            elif token.tag_ == 'TO':
                counts['tos'] += 1
        
        return counts
    
    def _count_capitals(self, text: str) -> int:
        """Count words that are fully capitalized (length > 1)."""
        words = text.split()
        return sum(1 for word in words if word.isupper() and len(word) > 1)
    
    def _count_short_words(self, words: List[str], max_length: int = 3) -> int:
        """Count words with length <= max_length."""
        return sum(1 for word in words if len(word) <= max_length)


def preprocess_text(text: str) -> str:
    """
    Preprocess text for the model (matching notebook preprocessing).
    
    Args:
        text: Raw input text.
    
    Returns:
        Cleaned and preprocessed text.
    """
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove @mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtag symbol (keep the word)
    text = re.sub(r'#', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_text_features(
    text: str, 
    user_features: Optional[Dict[str, float]] = None,
    as_array: bool = True
) -> np.ndarray | Dict[str, float]:
    """
    Quick function to extract features from text.
    
    Args:
        text: The input text to analyze.
        user_features: Optional dict with user profile features.
        as_array: If True, return numpy array; otherwise return dict.
    
    Returns:
        Features as numpy array or dictionary.
    
    Example:
        >>> features = extract_text_features("This is a FAKE news article! #fakenews")
        >>> print(features.shape)
        (20,)
    """
    extractor = TextFeatureExtractor()
    
    if as_array:
        return extractor.extract_features_array(text, user_features)
    return extractor.extract_features(text, user_features)


if __name__ == '__main__':
    sample_text = """
    BREAKING: Scientists discover new evidence! This is absolutely incredible news.
    Check out https://example.com for more info. #science #discovery @news_outlet
    The research was conducted by Dr. John Smith at Harvard University.
    """
    
    extractor = TextFeatureExtractor()
    features = extractor.extract_features(sample_text)
    
    print("Extracted Features:")
    print("-" * 50)
    for key, value in features.items():
        print(f"{key:25s}: {value}")
    
    print("\nFeature Array Shape:", extractor.extract_features_array(sample_text).shape)
