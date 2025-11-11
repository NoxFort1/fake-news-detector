import streamlit as st
import time
import re

import pickle
import numpy as np
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences

@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model('models/fake_news_model.h5')

        with open('models/tokenizer.pkl', 'rb') as file:
            tokenizer = pickle.load(file)

        with open('models/config.pkl', 'rb') as file:
            config = pickle.load(file)

        return model, tokenizer, config
    except Exception as e:
        st.error(f'Error loading model: {e}')
        return None, None, None

model, tokenizer, config = load_model()

def predict_article(text):
    if model is None or tokenizer is None:
        return None

    try:
        sequence = tokenizer.texts_to_sequences([text])

        vocabulary_size = config.get('vocabulary_size', 10000)
        sequence = [[token if token <= vocabulary_size else 0 for token in seq] for seq in sequence]

        padded = pad_sequences(
            sequence,
            maxlen = config['max_len'],
            padding = 'post',
            truncating = 'post'
        )

        probability = model.predict(padded, verbose = 0)[0][0]

        is_real = probability > 0.5
        confidence = probability if is_real else (1 - probability)

        return {
            'verdict': 'REAL NEWS' if is_real else 'FAKE NEWS',
            'confidence': float(confidence * 100),
            'real_prob': float(probability * 100),
            'fake_prob': float((1 - probability) * 100),
            'is_real': bool(is_real)
        }
    except Exception as e:
        st.error(f'Error during prediction: {e}')
        return None

def clear_page(placeholders):
    for placeholder in placeholders:
        placeholder.empty()

def display_progress(placeholder):
    placeholder.progress(0, '*Processing your text*')
    time.sleep(1)
    placeholder.progress(25, '*Processing your text*')
    time.sleep(1)
    placeholder.progress(50, '*Processing your text*')
    time.sleep(1)
    placeholder.progress(75, '*Processing your text*')
    time.sleep(1)
    placeholder.progress(100, '*Processing your text*')

    placeholder.empty()

def validate_url(url):
    if not url or not url.strip():
        return False, 'Please enter a URL'

    url = url.strip()

    if not url.startswith(('http://', 'https://')):
        return False, 'Please enter a valid URL starting with http:// or https://'

    pattern = r'^https?://[a-zA-Z0-9._-]+(\.[a-zA-Z0-9._-]+)+(:[0-9]+)?(/[^\s]*)?$'

    if not re.match(pattern, url):
        return False, 'Please enter a correct URL'

    return True, None

def show_results(placeholder, result):
    placeholder.markdown('### 📊 Analysis Results')

    if result['is_real']:
        placeholder.success(f"✅ **{result['verdict']}**")
    else:
        placeholder.error(f"🔴 **{result['verdict']}**")

    placeholder.metric(
        label = 'Model Confidence',
        value = f'{result['confidence']:.1f}%'
    )

    placeholder.markdown('#### Probability Breakdown:')
    col1, col2 = placeholder.columns(2)

    with col1:
        placeholder.metric('Real News', f'{result['real_prob']:.1f}%')
    with col2:
        placeholder.metric('Fake News', f'{result['fake_prob']:.1f}%')

    placeholder.progress(result['real_prob'] / 100)

    with placeholder.expander("ℹ️ About the Model"):
        placeholder.write("""
            **Model:** Bidirectional LSTM
            **Training Data:** ~38,000 articles
            **Accuracy:** ~95-97%
            **Max Length:** 350 tokens
            """)

def show_mock_results_url_file(placeholder):
    placeholder.markdown('### Analysis Results')

    col1, col2 = placeholder.columns(2)

    with col1:
        placeholder.metric('ML Model Prediction', 'FAKE', delta = '~85%')
    with col2:
        placeholder.metric('Gemini AI Analysis', 'FAKE', delta = '~85%')

    placeholder.info('URL and FILE analysis coming soon!')

if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'input_data' not in st.session_state:
    st.session_state.input_data = None


st.title("Fake News Detector")
st.markdown('*Analyze news articles using AI to detect potential misinformation*')
st.divider()

radio_placeholder = st.empty()
input_placeholder = st.empty()
button_placeholder = st.empty()
progress_placeholder = st.empty()
error_placeholder = st.empty()
results_placeholder = st.empty()

choice = radio_placeholder.radio('Choose how you want to verify the content of the article',
                  ['TEXT', 'URL', 'FILE'],
                  captions = ['Paste the full content', 'Use a URL', 'Attach a file'],
                  index = None)

if choice == 'TEXT':
    article_text = st.text_area('Enter your text', max_chars=50000, height=300)

    if st.button('🔎 Analyze Article', key='submit_text', type="primary"):
        if not article_text or not article_text.strip():
            st.error('Please enter a text')

        elif len(article_text.strip()) < 50:
            st.warning('⚠️ Text is too short. Please provide at least 50 characters.')

        else:
            st.empty()

            # Analyze
            with st.spinner('🔍 Analyzing article with LSTM model...'):
                result = predict_article(article_text)

            if result is not None:
                st.divider()
                st.markdown('### 📊 Analysis Results')

                if result['is_real']:
                    st.success(f"✅ **{result['verdict']}**")
                else:
                    st.error(f"🔴 **{result['verdict']}**")

                st.metric("Model Confidence", f"{result['confidence']:.1f}%")

                st.markdown("#### Probability Breakdown:")
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Real News", f"{result['real_prob']:.1f}%")

                with col2:
                    st.metric("Fake News", f"{result['fake_prob']:.1f}%")

                st.progress(float(result['real_prob']) / 100)

                with st.expander("ℹ️ About the Model"):
                    st.write("""
                    **Model:** Bidirectional LSTM
                    **Training Data:** ~38,000 articles
                    **Accuracy:** ~95-97%
                    **Max Length:** 350 tokens
                    """)

            else:
                st.error('❌ Error analyzing article. Please try again.')

if choice == 'URL':
    error_placeholder.empty()

    url = input_placeholder.text_input('Enter your URL', max_chars = 500,
                        placeholder = 'Example: https://www.bbc.com/news/article-123')

    if button_placeholder.button('Submit', key = 'submit_url'):
        is_valid, error_msg = validate_url(url)

        if is_valid:
            st.session_state.input_data = url

            clear_page([radio_placeholder, input_placeholder, button_placeholder, error_placeholder])

            display_progress(progress_placeholder)

            show_mock_results_url_file(results_placeholder)

            st.session_state.processed = True
        else:
            error_placeholder.error(error_msg)

if choice == 'FILE':
    error_placeholder.empty()

    uploaded_file = input_placeholder.file_uploader('Upload your file', type = ['txt', 'pdf', 'docx'])

    if button_placeholder.button('Submit', key = 'submit_file'):
        if uploaded_file:
            st.session_state.input_data = uploaded_file.name

            clear_page([radio_placeholder, input_placeholder, button_placeholder, error_placeholder])

            display_progress(progress_placeholder)

            show_mock_results_url_file(results_placeholder)

            st.session_state.processed = True
        else:
            error_placeholder.error('Please enter a file')