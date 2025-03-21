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

class RAGChatbot:
    def __init__(self, huggingface_token, weaviate_url, weaviate_api_key):
        login(token=huggingface_token)
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=Auth.api_key(weaviate_api_key),
        )
        self.model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
            self.load_chat_history()

    def load_chat_history(self):
        try:
            with open(LOG_FILE, "r") as f:
                st.session_state.chat_history = json.load(f)

            # Ensure every chat entry has a "date" field
            for chat in st.session_state.chat_history:
                if "date" not in chat:
                    chat["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        except (FileNotFoundError, json.JSONDecodeError):
            st.session_state.chat_history = []

    def save_chat_history(self):
        with open(LOG_FILE, "w") as f:
            json.dump(st.session_state.chat_history, f, indent=4)

        # Save chat history to CSV
        self.save_chat_to_csv()

    def save_chat_to_csv(self):
        """Save chat history to a CSV file."""
        if st.session_state.chat_history:
            df = pd.DataFrame(st.session_state.chat_history)
            df.to_csv(CSV_FILE, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)

    def search_documents(self, query):
        query_embedding = self.model.encode([query])
        documents = self.client.collections.get("TextChunks")
        response = documents.query.near_vector(near_vector=query_embedding[0])
        return [obj.properties["text"] for obj in response.objects]

    def generate_response(self, query, context):
        prompt = f"Context:\n{context}\n\nQuestion: {query}"
        response = ollama.chat(model="llama3.2:3b", messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]

    def run_chatbot(self):
        st.set_page_config(page_title="StartuPedia!", layout="wide")
        st.sidebar.title("Chat History")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display chat history in sidebar (latest first)
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            st.sidebar.write(f"{len(st.session_state.chat_history) - i}. {chat['user']}")

        st.title("StartuPedia")
        user_input = st.text_input("Ask a question:")

        if user_input:
            retrieved_docs = self.search_documents(user_input)
            context = "\n".join(retrieved_docs)
            response = self.generate_response(user_input, context)

            # Store chat history (latest message at the top)
            new_chat = {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user": user_input,
                "bot": response,
            }
            st.session_state.chat_history.insert(0, new_chat)  # Insert new chat at the top
            self.save_chat_history()

        # Display chat messages (latest first)
        st.subheader("Conversation")
        for chat in st.session_state.chat_history:
            date = chat.get("date", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))  # Handle missing date
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
