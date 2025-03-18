from sentence_transformers import SentenceTransformer
import streamlit as st
import weaviate
import ollama
import json
from weaviate.classes.init import Auth
from huggingface_hub import login

LOG_FILE = "chat_log.json"

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
        except (FileNotFoundError, json.JSONDecodeError):
            st.session_state.chat_history = []

    def save_chat_history(self):
        with open(LOG_FILE, "w") as f:
            json.dump(st.session_state.chat_history, f, indent=4)

    def search_documents(self, query):
        query_embedding = self.model.encode([query]).tolist()
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
        
        # Display chat history in sidebar
        for i, chat in enumerate(st.session_state.chat_history):
            st.sidebar.write(f"{i+1}. {chat['user']}")
        
        st.title("StartuPedia")
        user_input = st.text_input("Ask a question:")
        
        if user_input:
            retrieved_docs = self.search_documents(user_input)
            context = "\n".join(retrieved_docs)
            response = self.generate_response(user_input, context)
            
            # Store chat history
            st.session_state.chat_history.append({"user": user_input, "bot": response})
            self.save_chat_history()
            
        # Display chat messages
        st.subheader("Conversation")
        for chat in st.session_state.chat_history:
            st.markdown(f"**User:** {chat['user']}")
            st.markdown(f"**Bot:** {chat['bot']}")
            st.write("---")

if __name__ == "__main__":
    huggingface_token = ""  # Your Hugging Face token
    WEAVIATE_URL="https://s2umskuhqz23dftu3da19g.c0.asia-southeast1.gcp.weaviate.cloud"
    WEAVIATE_API_KEY="uT1gw7iS9ncKF14lDqDmByDXeWLXwa1rI76p"
    
    chatbot = RAGChatbot(huggingface_token, WEAVIATE_URL, WEAVIATE_API_KEY)
    chatbot.run_chatbot()
