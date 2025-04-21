from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from config import model, max_tokens
import streamlit as st
import asyncio

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

    template = PromptTemplate(
        template="""
            Please summarize the research paper title "{paper_name}", with the following specifications:
            Explaination Style: {style}
            Explaination Length: {length}

            if Explaination Length is "Short (1 paragraph)", then the summary should be concise and to the point in 30-100 words.
            if Explaination Length is "Medium (2-3 paragraphs)", then the summary should be a bit more detailed, but still concise in 100-250 words.
            if Explaination Length is "Long (3-5 paragraphs)", then the summary should be detailed and cover all important aspects of the paper in 250-500 words.
            if Explaination Length is "Detailed (5+ paragraphs)", then the summary should be very detailed and cover all important aspects of the paper in 500-1000 words.

            1. Mathematical details:
                - Include relevant mathematical equations present in the paper.
                - Explain the mathematical concepts using simple, intuitive code snippets where ever applicable.
            
            2. Analogies:
                - Use relatable analogies to simplify complex ideas.
            
            If certain information is not available in the paper, kindly respond with: "Insufficient information available", instead of guessing.
            Please ensure the summary is clear, accurate and aligned with the provided style and length.
        """,
        input_variables=["paper_name","style","length"],
        validate_template=True
    )

    # fill the placeholder with user inputs
    prompt = template.invoke({
        "paper_name": paper_name,
        "style": style,
        "length": length
    })
    return prompt
