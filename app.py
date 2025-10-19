import streamlit as st

st.title("Fake News Detector")

choice = st.radio('Choose how you want to verify the content of the article',
                  ['TEXT', 'URL', 'FILE'],
                  captions = ['Paste the full content', 'Use a URL', 'Attach a file'],
                  index = None)

if choice == 'TEXT':
    article_text = st.text_area('Enter your text', max_chars = 500)
if choice == 'URL':
    url = st.text_input('Enter your URL', max_chars = 100,
                        placeholder = 'Example: https://www.google.com')
if choice == 'FILE':
    uploaded_file = st.file_uploader('Upload your file', type = ['txt', 'pdf', 'docx'])