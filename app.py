import streamlit as st
import time
import re

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

def show_mock_results(placeholder):
    placeholder.markdown('### Analysis Results')

    col1, col2 = placeholder.columns(2)

    with col1:
        placeholder.metric('ML Model Prediction', 'FAKE', delta = '~85%')
    with col2:
        placeholder.metric('Gemini AI Analysis', 'FAKE', delta = '~85%')

    placeholder.info('Detailed analysis will appear here after model integration.')


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
    error_placeholder.empty()

    article_text = input_placeholder.text_area('Enter your text', max_chars = 50000, height = 300)

    if button_placeholder.button('Submit', key = 'submit_text'):
        if article_text and article_text.strip():
            st.session_state.input_data = article_text

            clear_page([radio_placeholder, input_placeholder, button_placeholder, error_placeholder])

            display_progress(progress_placeholder)

            show_mock_results(results_placeholder)

            st.session_state.processed = True
        else:
            error_placeholder.error('Please enter a text')

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

            show_mock_results(results_placeholder)

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

            show_mock_results(results_placeholder)

            st.session_state.processed = True
        else:
            error_placeholder.error('Please enter a file')