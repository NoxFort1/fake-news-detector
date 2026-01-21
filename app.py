"""
Fake News Detector - Main Streamlit Application

A web application for detecting fake news in tweets using a CNN + BiLSTM + MLP model
combined with Gemini AI for additional analysis.
"""

import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

from src.config import PAGE_CONFIG, SIDEBAR_CSS
from src.precheck import local_text_precheck
from src.gemini_client import agent_truthfulness_assessment, agent_technical_explanation
from src.predictor import load_model, predict_tweet
from src.ui_components import show_results, render_sidebar

st.set_page_config(**PAGE_CONFIG)
st.markdown(SIDEBAR_CSS, unsafe_allow_html = True)

env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path = env_path)

model, tokenizer, config = load_model()

if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'input_data' not in st.session_state:
    st.session_state.input_data = None

st.title("Fake News Detector")
st.markdown('*Analyze tweets using AI to detect potential misinformation*')

render_sidebar()

st.divider()

st.subheader("Paste tweet text")
article_text = st.text_area('Paste tweet text', max_chars = 5000, height = 200)

if st.button('🔎 Analyze Tweet', key = 'submit_text', type = "primary"):
    if not article_text or not article_text.strip():
        st.error('Please enter a text')
    elif len(article_text.strip()) < 10:
        st.warning('⚠️ Text is too short. Please provide at least 10 characters.')
    else:
        suitable, reason = local_text_precheck(article_text)

        with st.expander("🧪 Gemini Precheck"):
            if suitable:
                st.success(f"Text accepted for analysis. Reason: {reason}")
            else:
                st.warning(f"Text rejected. Reason: {reason}")

        if not suitable:
            st.warning("⚠️ Analysis stopped. Please provide a more suitable text.")
        else:
            col_left, col_right = st.columns(2)

            with col_left:
                st.subheader("🧠 Agent Assessment")
                with st.spinner('🤖 Agent is assessing the tweet...'):
                    agent_view = agent_truthfulness_assessment(article_text)
                if agent_view:
                    st.markdown(
                        f"**Verdict:** {agent_view['verdict']}  \n"
                        f"**Confidence:** {agent_view['confidence']}%"
                    )
                    if agent_view.get("bullets"):
                        st.write("\n".join([f"- {b}" for b in agent_view["bullets"]]))
                else:
                    st.info('Set `GOOGLE_API_KEY` to enable agent assessment.')

            with col_right:
                st.subheader("📊 Model Prediction")
                with st.spinner('🔍 Analyzing tweet with CNN+LSTM model...'):
                    result = predict_tweet(article_text, model, tokenizer, config)

                if result is not None:
                    show_results(st, result)
                else:
                    st.error('❌ Error analyzing tweet. Please try again.')

            if result is not None:
                st.divider()
                with st.expander("🧪 Technical Explanation (Model)"):
                    technical = agent_technical_explanation(
                        article_text,
                        result['verdict'],
                        result['confidence'],
                        result['real_prob'],
                        result['fake_prob'],
                    )
                    if technical:
                        st.write(technical)
                    else:
                        st.info('Set `GOOGLE_API_KEY` to enable technical explanation.')
