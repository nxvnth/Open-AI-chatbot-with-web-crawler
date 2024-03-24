import subprocess
import os
import json
import codecs
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

def start_node_crawler(crawler_dir='gpt-crawler', output_file='output-1.json', encoding='utf-8'):
    npm_path = 'C:/Program Files/nodejs/npm.cmd'  # Adjust as per your npm installation path
    crawler_full_path = os.path.join(os.getcwd(), crawler_dir)
    
    result = subprocess.run([npm_path, 'start'], cwd=crawler_full_path, capture_output=True, text=True)
    
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

def get_response(user_input, vector_store, chat_history):
    retriever_chain = get_context_retriever_chain(vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": chat_history,
        "input": user_input
    })
    
    return response['answer']

if __name__ == "__main__":
    print("Chat with websites")
    hi = input("type hi to start")
    
    chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]
    vector_store = None
    
    if hi:
        # Start the crawler and get the path to the output JSON file
        json_file_path = start_node_crawler()
        vector_store = get_vectorstore_from_json(json_file_path)

        while True:
            user_query = input("Type your message here (or type 'exit' to quit): ")
            if user_query == 'exit':
                break
            if user_query:
                response = get_response(user_query, vector_store, chat_history)
                chat_history.append(HumanMessage(content=user_query))
                chat_history.append(AIMessage(content=response))
                print(f"AI: {response}")
