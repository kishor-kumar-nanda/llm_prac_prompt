from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline
from dotenv import load_dotenv
import streamlit as st
# import torch

load_dotenv()

def load_model():
    with st.spinner("Loading model.. Please wait"):
        llm = HuggingFaceEndpoint(
            repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            # repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            task="text-generation"
        )
        # llm = HuggingFacePipeline.from_model_id(
        #     # model_id="mistralai/Mistral-7B-Instruct-v0.2",
        #     model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        #     task="text-generation",
        #     device=0 if torch.cuda.is_available() else -1,
        #     pipeline_kwargs={
        #         "max_new_tokens":512,
        #         "temperature":0.7
        #     }
        # )
    return ChatHuggingFace(llm=llm)

model = load_model()
st.success("Model loaded successfully. You may interact with the app..")

st.markdown("<h1 style='text-align: center;'>Research Tool</h1>",unsafe_allow_html=True)

user_input = st.text_input("Enter your prompt", placeholder="e.g., Summarize the Attention is all you need")

if st.button("Summarize"):
    if model is None:
        st.error("Model not loaded. Please check your setup and try again.")
    elif not user_input:
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating response..."):
            try:
                result = model.invoke(user_input)
                st.write("**Response:**")
                st.write(result.content)
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

