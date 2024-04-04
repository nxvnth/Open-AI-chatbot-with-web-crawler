import fitz  # PyMuPDF
import time
import subprocess
import os
import json
import tempfile
import codecs
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from werkzeug.utils import secure_filename

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

def start_node_crawler(crawler_dir='gpt-crawler', output_file='output-1.json', encoding='utf-8'):
    npm_path = 'C:/Program Files/nodejs/npm.cmd'  # Adjust as per your npm installation path
    crawler_full_path = os.path.join(os.getcwd(), crawler_dir)
    
    result = subprocess.run([npm_path, 'start'], cwd=crawler_full_path, capture_output=True, text=True)
    # Run the flush.py script using the Python executable from the virtual environment
    flush_script_path = os.path.join(crawler_full_path, 'flush.py')

    flush_result = subprocess.run(['python', flush_script_path], capture_output=True, text=True)

    if flush_result.returncode != 0:
        error_message = flush_result.stderr
        print(f"Error running flush.py: {error_message}")
        raise Exception(f"flush.py failed with error: {error_message}")
    else:
        print("removed all cache files")
    if result.returncode != 0:
        error_message = result.stderr
        print(f"Error running crawler: {error_message}")
        raise Exception(f"Crawler failed with error: {error_message}")
    
    return os.path.join(crawler_full_path, output_file)

def get_vectorstore_from_json(json_file):
    with codecs.open(json_file, 'r', 'utf-8-sig') as f:
        data = json.load(f)
    
    text_splitter = RecursiveCharacterTextSplitter()
    documents = []

    for entry in data:
        html_content = entry['html']
        cleaned_content = html_content.replace('\\n', ' ').strip()
        chunks = text_splitter.split_text(cleaned_content)
        documents.extend([Document(chunk) for chunk in chunks])

    attempt = 0
    max_attempts = 5  # Set a maximum number of attempts to avoid infinite loops
    success = False
    vector_store = Chroma(collection_name='v_db',persist_directory='vector_store',embedding_function = OpenAIEmbeddings(openai_api_key=openai_api_key))
    while attempt < max_attempts and not success:
        
            vector_store.add_documents(
                documents,
                embedding_function=OpenAIEmbeddings(
                    openai_api_key=openai_api_key,
                    show_progress_bar=True,
                ),
                persist_directory='vector_store',
                collection_name='v_db'
            )
            vector_store.persist()
            print("Vectors generated and saved to disk at:", 'vector_store')
            success = True
            return vector_store 


def get_vectorstore_from_pdfs(directory_path='uploaded_files'):
    """
    Extracts text from all PDF files in a directory, splits the texts into chunks, converts these into embeddings,
    and adds them to an existing Chroma vector store.
    """
    # List all PDF files in the specified directory
    pdf_files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]
    print(pdf_files)
    documents = []  # Initialize a list to store Document objects
    
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter()
    
    # Process each PDF file
    for pdf_file in pdf_files:
        doc_path = os.path.join(directory_path, pdf_file)
        doc = fitz.open(doc_path)
        text = ""
        for page in doc:
            text += page.get_text()
        print(text)
        # Split the PDF text into manageable chunks
        chunks = text_splitter.split_text(text)
        # Convert each chunk into a Document and add to the documents list
        documents.extend([Document(chunk) for chunk in chunks])
    print(chunks)
    print(f"Processing {len(pdf_files)} PDF files.")
    
   
    attempt = 0
    max_attempts = 5
    success = False
    print("1")
    vector_store = Chroma(collection_name='v_db',persist_directory='vector_store',embedding_function = OpenAIEmbeddings(openai_api_key=openai_api_key))
    print("2")
    while attempt < max_attempts and not success:
            
            # This time, we use 'add_documents' method of the Chroma vector store
            # to add new embeddings to an existing collection.
            vector_store.add_documents(
                documents,
                embedding_function=OpenAIEmbeddings(
                    openai_api_key=openai_api_key,
                    show_progress_bar=True,
                ),
                persist_directory='vector_store',
                collection_name='v_db'
            )
            vector_store.persist()

            print("PDF vectors added to the existing vector store.")
            success = True
            print("3")
            return vector_store
    
        

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    vector_store = Chroma(collection_name='v_db',persist_directory='vector_store',embedding_function = OpenAIEmbeddings(openai_api_key=openai_api_key))
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input, vector_store, chat_history):
    retriever_chain = get_context_retriever_chain(vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": chat_history,
        "input": user_input
    })
    
    return response['answer']
"""
if __name__ == "__main__":
    print("Chat with websites")
    hi = 1
    vector_store = None
    while hi in range(1,3):
        hi = int(input("type 1 to start crawling or 2 to start chatbot: "))
        if hi==1:
            # Start the crawler and get the path to the output JSON file
            json_file_path = start_node_crawler()
            vector_store = get_vectorstore_from_json("gpt-crawler\output-1.json")
            con = input("type y to continue to chatbot or n to exit: ") 
            if con == 'y':
                continue
            else:
                break


        elif hi == 2:

            chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]
            while True:
                user_query = input("Type your message here (type pdf to load the pdf or type 'exit' to quit): ")
                if user_query == 'exit':
                    break
                if user_query == 'pdf':
                    get_vectorstore_from_pdfs()

                if user_query:
                    response = get_response(user_query, 'vector_store', chat_history)
                    chat_history.append(HumanMessage(content=user_query))
                    chat_history.append(AIMessage(content=response))
                    print(f"AI: {response}")
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import threading

app = app = Flask(__name__, static_folder='front-end-/dist', static_url_path='/')
CORS(app)
UPLOAD_FOLDER = 'uploaded_files'
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploaded_files')
print(app.config['UPLOAD_FOLDER'])
# Make sure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Global variable to store the vector store and chat history state
vector_store = None
chat_history = []

@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/start_crawler', methods=['POST'])
def start_crawler():
    global vector_store
    # Assuming you handle the crawler path and output file name within the function
    json_file_path = start_node_crawler()
    vector_store = get_vectorstore_from_json(json_file_path)
    return jsonify({"message": "Crawler started and vector store created."}), 200

@app.route('/send_message', methods=['POST'])
def send_message():
    global chat_history, vector_store
    data = request.json
    user_input = data.get('message')
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    response = get_response(user_input, vector_store, chat_history)
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))
    return jsonify({"message": response}), 200


@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    # This will handle the uploaded file and process it
    # Make sure to validate the file type and handle any errors
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    
    # Ensure the UPLOAD_FOLDER exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    
    file_path =os.path.join('uploaded_files', filename)
    try:
        file.save(file_path)
        get_vectorstore_from_pdfs()  # Assuming it processes all files in the directory
        return jsonify({"message": "File uploaded successfully"}), 200
    
    except Exception as e:
        app.logger.error(f'Failed to save file: {e}')
        return jsonify({"error": str(e)}), 500
    # Process the saved file

@app.route('/submit_url', methods=['POST'])
def submit_url():
    data = request.json
    print("Received data:", data)  # Log to console
    url = data.get('url')
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    
    # Specify the directory where your Node.js project can access it
    temp_dir = "gpt-crawler"
    temp_file_path = os.path.join(temp_dir, 'crawler_url.txt')
    
    try:
        print("try")
        with open(temp_file_path, 'w') as file:
            file.write(url)
        json_file_path = start_node_crawler()
        get_vectorstore_from_json(json_file_path)
        # Optionally, start the crawler here or have it check the file independently
        return jsonify({"message": "URL saved successfully"}), 200
    except Exception as e:
        app.logger.error(f'Failed to save URL: {e}')
        return jsonify({"error": str(e)}), 500

# If you need to provide a way to retrieve chat history
@app.route('/chat_history', methods=['GET'])
def get_chat_history():
    global chat_history
    # Convert chat history to a serializable format
    history = [{"content": message.content, "sender": type(message).__name__} for message in chat_history]
    return jsonify(history), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)

