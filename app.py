# app.py

import os
import time
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser # <-- ADD THIS IMPORT

# Load environment variables from API.env file
load_dotenv('API.env')

# Caching the function to improve performance
@st.cache_data
def process_and_summarize_pdf(pdf_file_path, custom_prompt_text, chain_type, chunk_size, chunk_overlap):
    """
    Processes and generates a response from a PDF using a user-provided prompt and settings.
    """
    # 1. Instantiate LLM model
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_key not found. Please set it in your API.env file.")
        return None

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.3,
        google_api_key=api_key
    )

    # 2. Load and split the PDF using user-defined chunk settings
    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load_and_split(
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    )
    
    st.sidebar.info(f"The document has been split into **{len(docs)}** chunks.")

    # Conditional chain creation
    if chain_type == "stuff":
        from langchain.chains.summarize import load_summarize_chain
        prompt_template = custom_prompt_text + "\n\n{text}"
        prompt = PromptTemplate.from_template(prompt_template)
        chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
        result = chain.invoke({"input_documents": docs})
        return result['output_text']

    elif chain_type == "map_reduce":
        # --- UPDATED: MANUAL MAP_REDUCE LOGIC WITH MODERN LCEL SYNTAX ---
        
        # 1. Map Step: Create the LCEL chain and invoke it for each chunk
        map_prompt_template = "Summarize this portion of the document:\n\n{text}"
        map_prompt = PromptTemplate.from_template(map_prompt_template)
        # The new "pipe" syntax for creating a chain
        map_chain = map_prompt | llm | StrOutputParser()
        
        individual_summaries = []
        for i, doc in enumerate(docs):
            st.sidebar.write(f"Processing chunk {i+1}/{len(docs)}...")
            # Use .invoke() with a dictionary input
            summary = map_chain.invoke({"text": doc.page_content})
            individual_summaries.append(summary)
            time.sleep(4) 
        
        summary_docs = [Document(page_content=s) for s in individual_summaries]

        # 2. Reduce Step: Create the LCEL chain and invoke it on the combined summaries
        combine_prompt_template = f"Combine the following summaries into a final, coherent response based on this instruction: '{custom_prompt_text}'.\n\nSummaries:\n{{text}}"
        combine_prompt = PromptTemplate.from_template(combine_prompt_template)
        # The new "pipe" syntax for the reduce chain
        reduce_chain = combine_prompt | llm | StrOutputParser()

        # Manually "stuff" the summaries together before invoking the final chain
        combined_summaries = "\n\n".join([doc.page_content for doc in summary_docs])
        final_result = reduce_chain.invoke({"text": combined_summaries})
        
        return final_result

def main():
    st.set_page_config(page_title="AI Document Assistant", page_icon="📄", layout="wide")
    
    st.title("📄 AI Document Assistant")
    st.markdown("Interact with your documents! Upload a PDF, ask a question or provide an instruction, and let the AI assist you.")

    # Sidebar for Configuration
    st.sidebar.header("⚙️ Configuration")
    chain_type = st.sidebar.selectbox("Select Chain Type", ["stuff", "map_reduce"])
    
    if chain_type == "map_reduce":
        st.sidebar.warning(
            "The 'map_reduce' method respects the API rate limit by adding a 4-second delay between processing each chunk. This may be slow for large documents."
        )

    chunk_size = st.sidebar.slider("Chunk Size", min_value=500, max_value=20000, value=4000, step=100)
    chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=0, max_value=5000, value=400, step=50)

    # Main Page Layout
    custom_prompt = st.text_area("1. Enter your prompt or query:", height=150,
                                 placeholder="For example: 'Summarize the key findings of this document for a non-technical audience in five bullet points.'")
    
    uploaded_file = st.file_uploader("2. Upload your PDF file", type="pdf")

    if uploaded_file is not None:
        st.sidebar.empty()
        temp_file_path = os.path.join(".", "temp_uploaded_file.pdf")
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Generate Response", type="primary"):
            if not custom_prompt:
                st.warning("Please enter a prompt to continue.")
            else:
                with st.spinner("🧠 The AI is thinking... This might take a moment, especially with map_reduce."):
                    try:
                        response = process_and_summarize_pdf(temp_file_path, custom_prompt, chain_type, chunk_size, chunk_overlap)
                        if response:
                            st.subheader("AI Generated Response")
                            st.success(response)
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    # Informational Notes Section
    with st.expander("ℹ️ What do these settings mean?"):
        st.markdown("""
        - **Chunk Size:** This is the size of each piece the document is broken into for processing. Think of it as the number of characters on each "page" the AI reads at a time.
        
        - **Chunk Overlap:** This is the amount of text that is repeated between consecutive chunks. It helps the AI maintain context by ensuring that no crucial information is lost at the split between two chunks.
        
        - **Chain Type: `stuff` vs. `map_reduce`**
          - **`stuff`**: This is the simplest method. It "stuffs" all the document chunks into a single prompt and sends it to the AI once. It's fast and cheap but only works for smaller documents that fit within the AI's context limit.
          - **`map_reduce`**: This is a two-step method for larger documents. First, it summarizes each chunk individually (**Map** step). Then, it combines those individual summaries into a final, coherent summary (**Reduce** step). It's slower due to the built-in delay but can handle very large files without hitting API rate limits.
        """)

if __name__ == "__main__":
    main()

