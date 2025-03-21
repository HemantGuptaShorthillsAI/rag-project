from sentence_transformers import SentenceTransformer
import streamlit as st
import weaviate
import ollama
import os
import json
import pandas as pd
from datetime import datetime
from weaviate.classes.init import Auth
from huggingface_hub import login
from dotenv import load_dotenv
import csv

load_dotenv()

LOG_FILE = "../assets/chat_log.json"
CSV_FILE = "../assets/chat_log.csv"

class ChatModel:
    def __init__(self, model_name="llama3.2:3b"):
        self.model_name = model_name
    
    def generate_response(self, query, context):
        prompt = f"Context:\n{context}\n\nQuestion: {query}"
        response = ollama.chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]

class EmbeddingModel:
    def __init__(self, model_name="nomic-ai/nomic-embed-text-v1"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
    
    def encode(self, text):
        return self.model.encode([text])

class RAGChatbot:
    def __init__(self, huggingface_token, weaviate_url, weaviate_api_key, model_name="llama3.2:3b", embed_model_name="nomic-ai/nomic-embed-text-v1"):
        login(token=huggingface_token)
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=Auth.api_key(weaviate_api_key),
        )
        self.chat_model = ChatModel(model_name)  # Use the ChatModel class
        self.embed_model = EmbeddingModel(embed_model_name)  # Use the EmbeddingModel class
        
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
            self.load_chat_history()

    def load_chat_history(self):
        try:
            with open(LOG_FILE, "r") as f:
                st.session_state.chat_history = json.load(f)
            for chat in st.session_state.chat_history:
                if "date" not in chat:
                    chat["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        except (FileNotFoundError, json.JSONDecodeError):
            st.session_state.chat_history = []

    def save_chat_history(self):
        with open(LOG_FILE, "w") as f:
            json.dump(st.session_state.chat_history, f, indent=4)
        self.save_chat_to_csv()

    def save_chat_to_csv(self):
        if st.session_state.chat_history:
            df = pd.DataFrame(st.session_state.chat_history)
            df.to_csv(CSV_FILE, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)

    def search_documents(self, query):
        query_embedding = self.embed_model.encode(query)
        documents = self.client.collections.get("TextChunks")
        response = documents.query.near_vector(near_vector=query_embedding[0])
        return [obj.properties["text"] for obj in response.objects]

    def run_chatbot(self):
        st.set_page_config(page_title="StartuPedia!", layout="wide")
        st.sidebar.title("Chat History")
        st.sidebar.subheader("Select Model")
        model_name = st.sidebar.selectbox("Choose a chat model", ["llama3.2:3b", "gpt-4", "gemini-pro"])
        embed_model_name = st.sidebar.selectbox("Choose an embedding model", ["nomic-ai/nomic-embed-text-v1", "all-MiniLM-L6-v2"])
        
        self.chat_model = ChatModel(model_name)  # Update chat model dynamically
        self.embed_model = EmbeddingModel(embed_model_name)  # Update embedding model dynamically
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            st.sidebar.write(f"{len(st.session_state.chat_history) - i}. {chat['user']}")

        st.title("StartuPedia")
        user_input = st.text_input("Ask a question:")

        if user_input:
            retrieved_docs = self.search_documents(user_input)
            context = "\n".join(retrieved_docs)
            response = self.chat_model.generate_response(user_input, context)
            new_chat = {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user": user_input,
                "bot": response,
            }
            st.session_state.chat_history.insert(0, new_chat)
            self.save_chat_history()

        st.subheader("Conversation")
        for chat in st.session_state.chat_history:
            date = chat.get("date", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            st.markdown(f"**User:** {chat['user']}")
            st.markdown(f"**Bot:** {chat['bot']}")
            st.markdown(f"*Date:* {date}")
            st.write("---")

if __name__ == "__main__":
    huggingface_token = os.getenv("huggingface_token")
    WEAVIATE_URL = os.getenv("WEAVIATE_URL")
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
    chatbot = RAGChatbot(huggingface_token, WEAVIATE_URL, WEAVIATE_API_KEY)
    chatbot.run_chatbot()
