# pip install streamlit langchain lanchain-openai beautifulsoup4 python-dotenv chromadb

import subprocess
import os
import json
import codecs
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


load_dotenv()
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}
# Function to start the Node.js crawler
def start_node_crawler(crawler_dir='gpt-crawler', output_file='output-1.json', encoding='utf-8'):
    # Define the full path to the npm command and the working directory
    npm_path = 'C:/Program Files/nodejs/npm.cmd'  # Adjust as per your npm installation path
    crawler_full_path = os.path.join(os.getcwd(), crawler_dir)
    
    # Start the crawler using npm start without changing the working directory
    result = subprocess.run([npm_path, 'start'], cwd=crawler_full_path, capture_output=True, text=True)
    
    # Check for errors
    if result.returncode != 0:
        # Handle the error, log it, or raise an exception
        error_message = result.stderr
        print(f"Error running crawler: {error_message}")
        raise Exception(f"Crawler failed with error: {error_message}")
    
    # Assuming the crawler writes the output to the specified output file
    return os.path.join(crawler_full_path, output_file)


def get_vectorstore_from_json(json_file):
    # Load the JSON data from the file
    with codecs.open(json_file, 'r', 'utf-8-sig') as f:  # utf-8-sig handles BOM
        data = json.load(f)
    
    # Process each entry's 'html' content and split into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    documents = []

    for entry in data:
        html_content = entry['html']
        cleaned_content = html_content.replace('\\n', ' ').strip()
        chunks = text_splitter.split_text(cleaned_content)
        documents.extend([Document(chunk) for chunk in chunks])

    vector_store = Chroma.from_documents(documents, OpenAIEmbeddings())

    return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    
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

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })
    
    return response['answer']


if __name__ == "__main__":
    st.set_page_config(page_title="Chat with websites", page_icon="🤖")
    st.title("Chat with websites")

    with st.sidebar:
        st.header("Settings")
        website_url = st.text_input("Website URL")

    if website_url:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]
        
        if "vector_store" not in st.session_state:
            # Start the crawler and get the path to the output JSON file
            json_file_path = start_node_crawler()
            st.session_state.vector_store = get_vectorstore_from_json(json_file_path)


        # user input
        user_query = st.chat_input("Type your message here...")
        if user_query is not None and user_query != "":
            response = get_response(user_query)
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))
        
       

        # conversation
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)
