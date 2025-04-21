import utility_functions as uf
from dotenv import load_dotenv
import streamlit as st
import time

load_dotenv()

if 'model' not in st.session_state:
    try:
        model = uf.load_model()
        if model.get_name() == "ChatHuggingFace":
            # Success message shows for 2 seconds inside a placeholder
            placeholder = st.empty()
            placeholder.success("Model loaded successfully. You may interact with the app..")
            time.sleep(2)
            placeholder.empty()
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
else:
    model = st.session_state['model']

st.markdown("<h1 style='text-align: center;'>Research Paper Summarizer</h1>",unsafe_allow_html=True)
# user_input = st.text_input("Enter your prompt", placeholder="e.g., Summarize the Attention is all you need")
prompt = uf.get_prompt()

if st.button("Summarize",use_container_width=True):
    if 'model' not in st.session_state:
        st.error("Model not loaded. Please check your setup and try again.")
    else:
        model = st.session_state['model']
        with st.spinner("Generating response..."):
            try:
                result = model.invoke(prompt)
                st.write("**Response:**")
                print(result.content)
                st.write(result.content)
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

