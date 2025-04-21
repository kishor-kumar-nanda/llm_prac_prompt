from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import load_prompt
from config import model, max_tokens
from prompt_generatory import generate_prompt
import streamlit as st
import asyncio
import os

async def _load_model_async():
    llm = HuggingFaceEndpoint(
            repo_id=model,
            task="text-generation",
            max_new_tokens=max_tokens
        )
    return ChatHuggingFace(llm=llm)
    # return llm

# Loading model
def load_model():
    with st.spinner("Loading model.. Please wait"):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            model_instance = loop.run_until_complete(_load_model_async())
            st.session_state['model'] = model_instance
            loop.close()
            return model_instance
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            return None

# User picked research paper input
def get_user_inputs():
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            paper_name = st.selectbox(
                "Pick a research paper",
                ("BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (BERT)", "GPT-3: Language Models are Few-Shot learners", "Generative Adversarial Networks (GAN)"),
            )

        with col2:
            style_type = st.selectbox(
                "Pick a style of summary",
                ("Beginner-Friendly", "Important points","Mathematical", "Code-Oriented"),
            )

        with col3:
            paper_length = st.selectbox(
                "How would you want your output be?",
                ("Short (1 paragraph)", "Medium (2-3 paragraphs)", "Long (3-5 paragraphs)", "Detailed (5+ paragraphs)"),
            )
    return paper_name,style_type,paper_length

# crafting propmt with help of user picked inputs
def get_prompt():
    paper_name,style,length = get_user_inputs()

    cwd = os.getcwd()
    template_path = os.path.join(cwd, "template.json")
    if not os.path.exists(template_path):
        generate_prompt()
    
    # load the prompt template from the file
    template = load_prompt(template_path)
    
    # fill the placeholder with user inputs
    prompt = template.invoke({
        "paper_name": paper_name,
        "style": style,
        "length": length
    })
    return prompt
