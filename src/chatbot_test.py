import unittest
from unittest.mock import patch, MagicMock
from chatbot import RAGChatbot, ChatModel
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

class TestRAGChatbot(unittest.TestCase):
    
    @patch('chatbot.login')
    @patch('chatbot.weaviate.connect_to_weaviate_cloud')
    def setUp(self, MockWeaviateConnect, MockLogin):
        self.huggingface_token = os.getenv("huggingface_token")
        self.weaviate_url = os.getenv("WEAVIATE_URL")
        self.weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
        self.model_name = "llama3.2:3b"
        
        # Initialize ChatModel
        self.model_handler = ChatModel(self.model_name)
        self.model_handler.encode_text = MagicMock(return_value=[[0.1, 0.2, 0.3]])
        self.model_handler.chat = MagicMock(return_value="This is a response")
        
        self.chatbot = RAGChatbot(self.huggingface_token, self.weaviate_url, self.weaviate_api_key, self.model_handler)
        self.chatbot.client = MagicMock()
    
    def test_load_chat_history(self):
        with patch('builtins.open', unittest.mock.mock_open(read_data='[{"user": "Hello", "bot": "Hi"}]')):
            self.chatbot.load_chat_history()
            self.assertEqual(len(st.session_state.chat_history), 1)
            self.assertEqual(st.session_state.chat_history[0]['user'], "Hello")

    def test_save_chat_history(self):
        st.session_state.chat_history = [{"user": "Hello", "bot": "Hi"}]
        expected_output = '[\n    {\n        "user": "Hello",\n        "bot": "Hi"\n    }\n]'

        with patch('builtins.open', unittest.mock.mock_open()) as mock_file, patch('json.dump') as mock_json_dump:
            self.chatbot.save_chat_history()
            mock_json_dump.assert_called_once()  # Ensure json.dump is called


    def test_search_documents(self):
        self.chatbot.client.collections.get.return_value.query.near_vector.return_value.objects = [
            MagicMock(properties={"text": "Document 1"}),
            MagicMock(properties={"text": "Document 2"})
        ]
        results = self.chatbot.search_documents("test query")
        self.assertEqual(len(results), 2)
        self.assertIn("Document 1", results)
        self.assertIn("Document 2", results)

    def test_generate_response(self):
        response = self.model_handler.generate_response("test query", "test context")
        self.assertEqual(response, "This is a response")

    def test_empty_chat_history(self):
        with patch('builtins.open', unittest.mock.mock_open(read_data="[]")):
            self.chatbot.load_chat_history()
            self.assertEqual(len(st.session_state.chat_history), 0)

    def test_search_documents_no_results(self):
        self.chatbot.client.collections.get.return_value.query.near_vector.return_value.objects = []
        results = self.chatbot.search_documents("test query")
        self.assertEqual(len(results), 0)

    def test_generate_response_with_empty_context(self):
        response = self.model_handler.chat("test query", "")
        self.assertEqual(response, "This is a response")

    def test_chat_history_persistence(self):
        st.session_state.chat_history = [{"user": "Hi", "bot": "Hello"}]
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            self.chatbot.save_chat_history()
            handle = mock_file()
            self.assertTrue(handle.write.called)

    def test_invalid_json_in_chat_history(self):
        with patch('builtins.open', unittest.mock.mock_open(read_data="{invalid_json}")):
            self.chatbot.load_chat_history()
            self.assertEqual(st.session_state.chat_history, [])

if __name__ == "__main__":
    unittest.main()
