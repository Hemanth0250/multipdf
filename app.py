import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from translate import Translator
from langdetect import detect

# Add the necessary imports for text-to-speech and speech-to-text
import pyttsx3
import speech_recognition as sr

load_dotenv()
os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def translate_text(text, source_language, target_language):
    translator = Translator(from_lang=source_language, to_lang=target_language)
    translation = translator.translate(text)
    return translation

def user_input(user_question, conversation_history, source_language, target_language):
    # If the source language is not English, translate the question to English
    if source_language != "en":
        user_question_translated = translate_text(user_question, source_language, "en")
    else:
        user_question_translated = user_question

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Load FAISS index with dangerous deserialization enabled
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question_translated)

    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents": docs, "question": user_question_translated},
        return_only_outputs=True
    )

    # Translate the response back to the target language
    if target_language != "en":
        response_translated = translate_text(response["output_text"], "en", target_language)
    else:
        response_translated = response["output_text"]

    conversation_history.append({"question": user_question, "response": response_translated})
    return response_translated

def main():
    st.set_page_config("Chat PDF")
    st.header("Docu Detective.ai💁")

    user_question = st.text_input("Ask a Question from the PDF Files")
    conversation_history = st.session_state.get("conversation_history", [])

    st.subheader("Select Languages:")
    col1, col2 = st.columns(2)
    with col1:
        source_language = st.radio("Source Language:", options=["en", "fr", "es"], key="source_language")
    with col2:
        target_language = st.radio("Target Language:", options=["en", "fr", "es"], key="target_language")

    if user_question:
        response = user_input(user_question, conversation_history, source_language, target_language)
        st.write("Reply: ", response)

    # Add the text-to-speech and speech-to-text buttons
    st.subheader("Additional Options:")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Text-to-Speech", key="text_to_speech"):
            # Perform text-to-speech
            engine = pyttsx3.init()
            engine.say(response)
            engine.runAndWait()
        st.markdown('<style>div.stButton>button {background-color: #4CAF50; color: white; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 8px; width: 200px;}</style>', unsafe_allow_html=True)

    with col2:
        if st.button("Speech-to-Text", key="speech_to_text"):
            # Perform speech-to-text
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                st.write("Speak now...")
                audio = recognizer.listen(source)
            try:
                text = recognizer.recognize_google(audio)
                st.write("You said:", text)
            except sr.UnknownValueError:
                st.write("Sorry, could not understand audio.")
            except sr.RequestError as e:
                st.write("Error:", e)
        st.markdown('<style>div.stButton>button {background-color: #008CBA; color: white; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 8px; width: 200px;}</style>', unsafe_allow_html=True)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    # Display conversation history
    st.subheader("Conversation History")
    for item in conversation_history:
        st.write("Question: ", item["question"])
        st.write("Response: ", item["response"])
        st.write("---")
    st.session_state.conversation_history = conversation_history

if __name__ == "__main__":
    main()
