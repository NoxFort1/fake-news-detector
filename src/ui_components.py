"""UI components and rendering functions."""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from src.config import MODEL_METRICS, DATASET_DISTRIBUTION, CONFUSION_MATRIX, COLORS


def show_results(placeholder, result: dict):
    """
    Display prediction results in the given placeholder.
    
    Args:
        placeholder: Streamlit container to render in
        result: Dictionary with prediction results
    """
    placeholder.markdown('### 📊 Analysis Results')

    if result['is_real']:
        placeholder.success(f"✅ **{result['verdict']}**")
    else:
        placeholder.error(f"🔴 **{result['verdict']}**")

    placeholder.metric(
        label='Model Confidence',
        value=f"{result['confidence']:.1f}%"
    )

    placeholder.markdown('#### Probability Breakdown:')
    col1, col2 = placeholder.columns(2)

    with col1:
        placeholder.metric('Real', f"{result['real_prob']:.1f}%")
    with col2:
        placeholder.metric('Fake', f"{result['fake_prob']:.1f}%")

    placeholder.progress(result['real_prob'] / 100)


def render_sidebar():
    """Render the sidebar with model information and charts."""
    with st.sidebar:
        st.header("ℹ️ About the Model")
        
        st.markdown("""
        **Architecture:** CNN + BiLSTM + MLP
        
        The model combines two input branches:
        - **Text branch:** CNN + BiLSTM layers process tokenized tweet text
        - **Metadata branch:** Linguistic features (word counts, POS tags, named entities, punctuation patterns)
        
        Both branches are concatenated and passed through dense layers for classification.
        
        **Training Data:** 134,198 tweets  
        **Max Length:** 60 tokens
        """)
        
        st.divider()
        st.subheader("📊 Model Performance")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{MODEL_METRICS['Accuracy']:.2%}")
            st.metric("Precision", f"{MODEL_METRICS['Precision']:.2%}")
            st.metric("ROC-AUC", f"{MODEL_METRICS['ROC-AUC']:.2%}")
        with col2:
            st.metric("Recall", f"{MODEL_METRICS['Recall']:.2%}")
            st.metric("F1-Score", f"{MODEL_METRICS['F1-Score']:.2%}")
        
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(MODEL_METRICS.keys(), MODEL_METRICS.values(), color = COLORS['metrics'], alpha = 0.8)
        ax.set_ylim(0.95, 1.0)
        ax.set_ylabel('Score')
        ax.set_title('Performance Metrics')
        ax.grid(axis = 'y', alpha = 0.3)
        for bar, val in zip(bars, MODEL_METRICS.values()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002, 
                    f'{val:.2%}', ha = 'center', va = 'bottom', fontsize = 8)
        plt.xticks(rotation = 45, ha = 'right', fontsize = 8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.divider()
        st.subheader("📈 Dataset Distribution")
        
        fig2, ax2 = plt.subplots(figsize = (5, 4))
        ax2.pie(DATASET_DISTRIBUTION['sizes'], labels = DATASET_DISTRIBUTION['labels'], 
                autopct = '%1.1f%%', colors = COLORS['pie'], startangle = 90)
        ax2.set_title('Training Data Distribution')
        st.pyplot(fig2)
        plt.close()
        
        st.divider()
        st.subheader("🎯 Confusion Matrix")
        
        fig3, ax3 = plt.subplots(figsize = (5, 4))
        im = ax3.imshow(CONFUSION_MATRIX, interpolation = 'nearest', cmap = 'Blues')
        ax3.set_xticks([0, 1])
        ax3.set_yticks([0, 1])
        ax3.set_xticklabels(['FAKE', 'REAL'])
        ax3.set_yticklabels(['FAKE', 'REAL'])
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        ax3.set_title('Test Set Confusion Matrix')
        
        for i in range(2):
            for j in range(2):
                ax3.text(j, i, CONFUSION_MATRIX[i, j], ha = 'center', va = 'center', 
                        color = 'white' if CONFUSION_MATRIX[i, j] > 5000 else 'black', fontsize = 12)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()
