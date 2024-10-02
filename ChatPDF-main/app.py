# importing dependencies
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import faiss
from langchain.prompts import PromptTemplate
from htmlTemplates import css, bot_template, user_template
from groq import Groq

# Initialize Groq client with API key
groq = Groq(api_key="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")

# Creating a custom template to guide the LLM model
custom_template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

# Function to extract text from PDFs
def get_pdf_text(docs):
    text = ""
    for pdf in docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Avoid NoneType errors if extract_text fails
    return text

# Function to get response from Groq LLM
def get_groq_response(user_query, relevant_context):
    try:
        # Combine user query with relevant context from the vector store
        combined_input = f"{relevant_context}\n\n{user_query}"
        
        # Send request to Groq client
        completion = groq.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an AI assistant. Always respond in JSON format."},
                {"role": "user", "content": combined_input}
            ],
            model="llama3-8b-8192",
            temperature=0.3,
            max_tokens=3400,
            top_p=1,
            stream=False,
            response_format={"type": "json_object"},
            seed=888,
        )
        # Extract and return response
        response = completion.choices[0].message.content
        return response
    except Exception as e:
        st.error(f"Error querying the classifier: {e}")
        return None

# Function to split text into chunks
def get_chunks(raw_text):
    text_splitter = CharacterTextSplitter(separator="\n",
                                          chunk_size=1000,
                                          chunk_overlap=200,
                                          length_function=len)
    chunks = text_splitter.split_text(raw_text)
    return chunks

# Function to create a vector store using text chunks
def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    vectorstore = faiss.FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

# Function to handle user queries and display responses
def handle_question(question):
    # Retrieve relevant chunks based on the user's question
    if "vectorstore" in st.session_state:
        relevant_contexts = st.session_state.vectorstore.similarity_search(question, k=3)  # Get top 3 relevant chunks
        relevant_context = "\n".join([context.page_content for context in relevant_contexts])  # Combine relevant contexts
        
        response_content = get_groq_response(question, relevant_context)
        
        if response_content:
            st.write(bot_template.replace("{{MSG}}", response_content), unsafe_allow_html=True)
            # Update chat history
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []  # Initialize if it doesn't exist
            st.session_state.chat_history.append({"role": "assistant", "content": response_content})
        else:
            st.error("Failed to get a response from Groq")
    else:
        st.error("No documents processed yet. Please upload and process a PDF.")

# Main function for Streamlit app
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Initialize session state variables
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # Initialize as an empty list
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None  # Initialize vectorstore as None

    st.header("Chat with multiple PDFs :books:")
    question = st.text_input("Ask a question from your document:")
    if question:
        handle_question(question)

    with st.sidebar:
        st.subheader("Your documents")
        docs = st.file_uploader("Upload your PDF here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # Extract text from the PDFs
                raw_text = get_pdf_text(docs)
                
                # Convert the extracted text into chunks
                text_chunks = get_chunks(raw_text)

                # Create a vector store from the text chunks
                vectorstore = get_vectorstore(text_chunks)

                # Store the vectorstore in session state for further use
                st.session_state.vectorstore = vectorstore
                st.success("Documents processed successfully!")

if __name__ == '__main__':
    main()
