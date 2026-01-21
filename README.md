# 🔍 Fake News Detector

A web application for detecting fake news in tweets using a **CNN + BiLSTM + MLP** deep learning model combined with **Google Gemini AI** for additional analysis.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)

</div>

## 📋 Overview

This project implements a hybrid approach to fake news detection:
- **Deep Learning Model**: CNN + BiLSTM + MLP architecture trained on tweet data
- **LLM Agent**: Google Gemini 2.0 Flash for contextual truthfulness assessment
- **Feature Extraction**: Linguistic and statistical features using spaCy NLP

## 📁 Data

The dataset is not included in this repository. <br>
Download it from [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/truthseeker-2023.html) <br>
and place the CSV files in `data/raw/` folder.

## 🎯 Model Performance

<div align="center">

| Metric | Score |
|--------|-------|
| Accuracy | **98.03%** |
| Precision | **98.50%** |
| Recall | **97.66%** |
| F1-Score | **98.08%** |
| ROC-AUC | **99.79%** |

</div>

## ✨ Features

- **Dual Analysis**: Combines neural network predictions with AI-powered assessment
- **Text Feature Extraction**: 20+ linguistic features including:
  - Named Entity Recognition (NORP, PERSON, MONEY, CARDINAL)
  - Part-of-Speech analysis (verbs, adjectives, pronouns)
  - Social media signals (hashtags, URLs, mentions)
  - Statistical measures (word count, capitals, exclamations)
- **Technical Explanations**: AI-generated explanations for model predictions
- **Interactive Web UI**: Built with Streamlit for easy access

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/NoxFort1/fake-news-detector.git
```
Windows
```bash
cd fake-news-detector
```
Linux/MacOS
```bash
source venv/bin/activate
```
### 2. Create virtual environment
```python
python -m venv venv
```
Windows
```bash
venv\Scripts\activate
```
Linux/macOS
```bash
source venv/bin/activate
```
### 3. Install dependencies
```python
pip install -r requirements.txt
```
### 4. Download spaCy model
```python
python -m spacy download en_core_web_sm
```
### 5. (Optional) Configure Gemini API
Create a .env file in the project root:
```python
GOOGLE_API_KEY=your_google_api_key_here
```

## 💻 Usage

Run the web application
```bash
streamlit run app.py
```
The application will open in your browser at `http://localhost:8501` </br>
### How to use
1. Paste a tweet text in the input area
2. Click `🔎 Analyze Tweet`
3. View results from both the ML model and AI agent
4. Expand "Technical Explanation" for detailed analysis

## 📁 Project Structure
```
fake-news-detector/
├── app.py                        # Main Streamlit application
├── text_features.py              # Text feature extraction module
├── requirements.txt              # Python dependencies
├── models/
│   ├── cnn_lstm_mlp_model.keras  # Trained model
│   ├── tokenizer.pkl             # Text tokenizer
│   └── config.pkl                # Model configuration
├── src/
│   ├── config.py                 # App configuration & constants
│   ├── gemini_client.py          # Gemini API integration
│   ├── predictor.py              # Model loading & prediction
│   ├── precheck.py               # Input validation
│   └── ui_components.py          # Streamlit UI components
└── data/
    └── raw/                      # Training datasets
```
## 🛠️ Technologies Used

- Deep Learning: TensorFlow/Keras
- NLP: spaCy, Transformers
- LLM: Google Generative AI (Gemini 2.0 Flash)
- Web Framework: Streamlit
- Data Processing: NumPy, Pandas, scikit-learn
- Visualization: Matplotlib, Seaborn, WordCloud

## 📊 Model Architecture

The model uses a hybrid architecture:
1. CNN Layer: Captures local n-gram patterns
2. BiLSTM Layer: Captures sequential dependencies
3. MLP Head: Combines text embeddings with metadata features
4. Output: Binary classification (REAL/FAKE)

## 📝 License
This project is for educational purposes.

## 👤 Authors
Bartosz Sychowicz, Kamil Sitko


