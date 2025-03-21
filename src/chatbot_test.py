import unittest
from unittest.mock import patch, MagicMock
from chatbot import RAGChatbot
import streamlit as st
from dotenv import load_dotenv
import os
load_dotenv()

class Testchatbot(unittest.TestCase):

    @patch('chatbot.login')
    @patch('chatbot.weaviate.connect_to_weaviate_cloud')
    @patch('chatbot.SentenceTransformer')
    def setUp(self, MockSentenceTransformer, MockWeaviateConnect, MockLogin):
        self.huggingface_token = os.getenv("huggingface_token")
        self.weaviate_url = os.getenv("WEAVIATE_URL")
        self.weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
        self.chatbot = RAGChatbot(self.huggingface_token, self.weaviate_url, self.weaviate_api_key)
        self.chatbot.client = MagicMock()
        self.chatbot.model = MagicMock()

    def test_load_chat_history(self):
        with patch('builtins.open', unittest.mock.mock_open(read_data='[{"user": "Hello", "bot": "Hi"}]')):
            self.chatbot.load_chat_history()
            self.assertEqual(len(st.session_state.chat_history), 1)
            self.assertEqual(st.session_state.chat_history[0]['user'], "Hello")

    def test_save_chat_history(self):
        st.session_state.chat_history = [{"user": "Hello", "bot": "Hi"}]
        expected_output = '[\n    {\n        "user": "Hello",\n        "bot": "Hi"\n    }\n]'
        
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            self.chatbot.save_chat_history()
            handle = mock_file()
            written_content = "".join(call.args[0] for call in handle.write.call_args_list)
            self.assertEqual(written_content, expected_output)

    def test_search_documents(self):
        self.chatbot.model.encode.return_value = [[0.1, 0.2, 0.3]]
        self.chatbot.client.collections.get.return_value.query.near_vector.return_value.objects = [
            MagicMock(properties={"text": "Document 1"}),
            MagicMock(properties={"text": "Document 2"})
        ]
        results = self.chatbot.search_documents("test query")
        self.assertEqual(len(results), 2)
        self.assertIn("Document 1", results)
        self.assertIn("Document 2", results)

    @patch('chatbot.ollama.chat')
    def test_generate_response(self, MockOllamaChat):
        MockOllamaChat.return_value = {"message": {"content": "This is a response"}}
        response = self.chatbot.generate_response("test query", "test context")
        self.assertEqual(response, "This is a response")

    def test_empty_chat_history(self):
        with patch('builtins.open', unittest.mock.mock_open(read_data="[]")):
            self.chatbot.load_chat_history()
            self.assertEqual(len(st.session_state.chat_history), 0)

    def test_search_documents_no_results(self):
        self.chatbot.model.encode.return_value = [[0.1, 0.2, 0.3]]
        self.chatbot.client.collections.get.return_value.query.near_vector.return_value.objects = []
        results = self.chatbot.search_documents("test query")
        self.assertEqual(len(results), 0)

    def test_generate_response_with_empty_context(self):
        with patch('chatbot.ollama.chat') as MockOllamaChat:
            MockOllamaChat.return_value = {"message": {"content": "This is a response"}}
            response = self.chatbot.generate_response("test query", "")
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

