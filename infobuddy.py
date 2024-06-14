# Importing traceback for handling exceptions
import traceback

# Importing Streamlit for creating web applications
import streamlit as st

# Importing PDF related modules and classes
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator
import pdfplumber

# Importing OS module for interacting with the operating system
import os

# Importing text processing modules and classes
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# Importing modules and classes related to vector stores
from langchain.vectorstores import FAISS

# Importing conversational AI modules and classes
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

# Importing Pillow and pillow_heif for image processing
from PIL import Image as PILImage
from pillow_heif import register_heif_opener

# Importing tempfile for creating temporary files
import tempfile


# Set Hugging Face Hub API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Function to read PDF files and extract text
def read_pdf(uploaded_files):
    all_text = ""
    try:
        for uploaded_file in uploaded_files:
            # Informing the user about the file being processed
            st.write(f"Step 1: Processing file: {uploaded_file.name}")

            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    all_text += page.extract_text()

            # Informing the user that text extraction is complete
            st.write(f"Step 2: Text extraction completed for {uploaded_file.name}")
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        traceback.print_exc()
        all_text = None
    return all_text

# Function to create index and setup conversational retrieval chain
def setup_qa_chain(uploaded_files, model_name, temperature):
    try:
        for uploaded_file in uploaded_files:
            # Inform the user about the model being set up
            st.write(f"Step 3: Setting up AI model: {model_name}")

            # Create a temporary file from the uploaded file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_file_path = temp_file.name

            # Creating loaders
            st.write(f"Step 4: Creating loaders for file: {uploaded_file.name}")
            loaders = [UnstructuredPDFLoader(temp_file_path)]
            st.write(f"Step 5: Loaders for file have been created, loaders can be used effectively: {uploaded_file.name}")

            # Creating vector store index
            st.write(f"Step 6: Creating vector store index for file: {uploaded_file.name}")
            index = VectorstoreIndexCreator(
                embedding=HuggingFaceEmbeddings(),
                text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            ).from_loaders(loaders)
            st.write(f"Step 7: From loaders Using HuggingFaceEmbeddings, CharacterTextSplitter created a VectorstoreIndexCreator: {uploaded_file.name}")

            # Setting up language model
            st.write(f"Step 8: LLM is being initiated: {uploaded_file.name}")
            llm = HuggingFaceHub(repo_id=model_name, model_kwargs={"temperature": temperature, "max_length": 50, "max_new_tokens": 10})
            st.write(f"Step 9: LLM initiation is completed using HuggingFaceHub, LLM is being used: {uploaded_file.name}")

            # Creating conversational retrieval chain
            st.write(f"Step 10: Creating conversational retrieval chain for file: {uploaded_file.name}")
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=index.vectorstore.as_retriever(),
                input_key="question"
            )
            st.write(f"Step 11: Conversational retrieval chain for file has been created, perfectly set to use: {uploaded_file.name}")

            # Informing the user that the AI model setup is complete.
            st.write(f"Step 12: AI model setup complete for {uploaded_file.name}")

        return chain
    except Exception as e:
        st.error(f"Error setting up conversational retrieval chain: {e}")
        traceback.print_exc()
        return None

# Function to handle user input and generate response
def generate_response(user_question, chain, chat_memory):
    try:
        if chain is not None:
            # Informing the user that the AI model is generating a response
            st.write("Generating response...")

            response = chain.run(user_question)
            chat_memory.append((user_question, response))

            # Informing the user that the response has been generated
            st.write("Response generated.")

            return response
        else:
            return "Error: Conversational retrieval chain is not set up properly. Please try again later."
    except Exception as e:
        st.error(f"Error generating response: {e}")
        traceback.print_exc()
        return "Error: Failed to generate response. Please try again later."

# Streamlit app
def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("InfoBuddy")

    # Initialize chat memory
    if 'chat_memory' not in st.session_state:
        st.session_state.chat_memory = []

    # File uploader to upload PDF files
    pdf_files = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)

    # Model selection dropdown
    model_options = {
        "T5 Base": "t5-base",
        "GPT2": "gpt2",
        "DialoGPT Small": "microsoft/DialoGPT-small",
        "GPT2 Medium": "openai-community/gpt2-medium",
        "GPT2 Large": "openai-community/gpt2-large",
        "GPT2 XL": "openai-community/gpt2-xl"
    }
    selected_model = st.selectbox("Select AI Model", list(model_options.keys()))

    if pdf_files:
        # Process PDF files and setup conversational retrieval chain
        pdf_text = read_pdf(pdf_files)
        if pdf_text is not None:
            # Ask user for a question
            user_question = st.text_input("Ask a Question")

            if user_question:
                model_name = model_options[selected_model]
                temperature = st.slider("Select Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

                # Informing the user about the question being processed
                st.write("Processing user question...")

                # Generate response based on user question
                chain = setup_qa_chain(pdf_files, model_name, temperature)
                # Informing the user about the temperature selection
                st.write(f"Selected temperature: {temperature}")

                response = generate_response(user_question, chain, st.session_state.chat_memory)
                # Split the response by paragraphs
                paragraphs = response.split('\n\n')

                # Print the first paragraph
                if paragraphs:
                    st.write("Response:", paragraphs[0])
                else:
                    st.write("No paragraphs found in the response.")

    # Display chat memory
    st.sidebar.header("Chat Memory")
    for i, (question, answer) in enumerate(st.session_state.chat_memory):
        st.sidebar.write(f"Question {i + 1}: {question}")
        st.sidebar.write(f"Answer {i + 1}: {answer}")
        st.sidebar.write("---")

if __name__ == "__main__":
    main()
