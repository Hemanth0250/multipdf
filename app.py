import tempfile
import streamlit as st
from PyPDF2 import PdfReader
import docx
import os
import pandas as pd
from pdf2image import convert_from_path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from translate import Translator
import pyttsx3
import speech_recognition as sr
import googlesearch
from PIL import Image
import pytesseract

# Specify the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

load_dotenv()
os.getenv("GOOGLE_API_KEY")

def extract_text_from_pdf(pdf_file):
    text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(pdf_file.read())
        temp_file_path = temp_file.name

    pdf_reader = PdfReader(temp_file_path)
    
    for page in pdf_reader.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text
        else:
            images = convert_from_path(temp_file_path, 500, poppler_path=r'C:\Users\Pranav\poppler-24.02.0\Library\bin')
            all_text = []
            for i, image in enumerate(images):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as img_file:
                    image.save(img_file.name, 'JPEG')
                    all_text.append(extract_text_from_image(img_file.name))
            text = '\n'.join(all_text)
    
    if not text:
        raise Exception("No text extracted from the PDF file")
    
    os.remove(temp_file_path)
    return text

def extract_text_from_word(word_file):
    text = ""
    doc = docx.Document(word_file)
    for para in doc.paragraphs:
        text += para.text
    print("Extracted text from Word:", text)
    return text

def extract_text_from_excel(excel_file):
    text = ""
    df = pd.read_excel(excel_file)
    for column in df.columns:
        for cell in df[column]:
            text += str(cell) + " "
    print("Extracted text from Excel:", text)
    return text

def extract_text_from_image(image_file):
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    print("Extracted text from Image:", text)
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Print or log the text chunks for debugging
    print("Text chunks:", text_chunks)
    
    try:
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except IndexError as e:
        st.error(f"IndexError occurred: {e}")
        st.error("Error: Failed to create vector store. Please check your input.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.error("Error: Failed to create vector store.")

def get_vector_store_with_error_handling(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        # Add logging to print out text chunks for debugging
        print("Text chunks:", text_chunks)
        
        # Check if text_chunks is empty
        if not text_chunks:
            raise ValueError("Input text chunks are empty")

        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except ValueError as ve:
        st.error(f"ValueError occurred: {ve}")
        st.error("Error: Input text chunks are empty.")
    except IndexError as e:
        st.error(f"IndexError occurred: {e}")
        st.error("Error: Failed to create vector store. Please check your input.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.error("Error: Failed to create vector store.")

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
    if source_language != "en":
        user_question_translated = translate_text(user_question, source_language, "en")
    else:
        user_question_translated = user_question

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question_translated)

    chain = get_conversational_chain()
    
    response = chain.invoke(
        {"input_documents": docs, "question": user_question_translated},
        return_only_outputs=True
    )

    if target_language != "en":
        response_translated = translate_text(response["output_text"], "en", target_language)
    else:
        response_translated = response["output_text"]

    if "answer is not available in the context" in response_translated.lower():
        # Fallback to web search
        search_results = list(googlesearch.search(user_question, num=1, stop=1))
        if search_results:
            response_translated = "I couldn't find the answer in the document. Here's what I found online:\n"
            response_translated += search_results[0]
        else:
            response_translated = "I couldn't find the answer in the document or online."

    conversation_history.append({"question": user_question, "response": response_translated})
    return response_translated

def main():
    st.set_page_config(page_title="Chat Docs")
    st.title("Docu Detective.ai💁")

    user_question = st.text_input("Ask a Question from the Documents")
    conversation_history = st.session_state.get("conversation_history", [])

    with st.sidebar:
        st.title("Menu:")
        documents = st.file_uploader("Upload your Documents", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = ""
                for doc_file in documents:
                    if doc_file.name.endswith('.pdf'):
                        raw_text += extract_text_from_pdf(doc_file)
                    elif doc_file.name.endswith('.docx'):
                        raw_text += extract_text_from_word(doc_file)
                    elif doc_file.name.endswith('.txt'):
                        raw_text += doc_file.getvalue().decode("utf-8")
                    elif doc_file.name.endswith('.xlsx'):
                        raw_text += extract_text_from_excel(doc_file)
                    elif doc_file.name.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                        raw_text += extract_text_from_image(doc_file)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store_with_error_handling(text_chunks)
                st.success("Done")

        # Preview button below process button
        if st.button("Preview Documents"):
            for doc_file in documents:
                if doc_file.name.endswith('.pdf'):
                    st.subheader("PDF Content:")
                    st.write(extract_text_from_pdf(doc_file))
                elif doc_file.name.endswith('.docx'):
                    st.subheader("DOCX Content:")
                    st.write(extract_text_from_word(doc_file))
                elif doc_file.name.endswith('.txt'):
                    st.subheader("TXT Content:")
                    st.write(doc_file.getvalue().decode("utf-8"))
                elif doc_file.name.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    st.subheader("Image Content:")
                    st.write(extract_text_from_image(doc_file))

        st.header("Languages:")
        source_language = st.radio("Source Language:", options=["en", "fr", "es", "hi"], key="source_language")
        target_language = st.radio("Target Language:", options=["en", "fr", "es", "hi"], key="target_language")

    if user_question:
        st.markdown("---")
        st.write("User: ", user_question)

        response = user_input(user_question, conversation_history, source_language, target_language)

        st.write("Reply: ", response)
        st.markdown("---")

    with st.sidebar:
        st.subheader("Additional Options:")
        if st.button("Text-to-Speech", key="text_to_speech"):
            engine = pyttsx3.init()
            engine.say(response)
            engine.runAndWait()

        if st.button("Speech-to-Text", key="speech_to_text"):
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

    st.subheader("Conversation History")
    for item in conversation_history:
        st.write("Question: ", item["question"])
        st.write("Response: ", item["response"])
        st.write("---")
    st.session_state.conversation_history = conversation_history

if _name_ == "_main_":
    main()
