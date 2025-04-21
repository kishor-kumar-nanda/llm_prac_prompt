from langchain_core.prompts import PromptTemplate

def generate_prompt():
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

        template.save("template.json")