import unittest
from unittest.mock import patch, MagicMock
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import weaviate
import ollama
from weaviate.classes.init import Auth
from sentence_transformers import util
import json

from evaluate import RAGEvaluator  # Replace 'evaluate' with the actual module name

class TestRAGEvaluator(unittest.TestCase):
    
    @patch("evaluate.weaviate.connect_to_weaviate_cloud")
    @patch("evaluate.SentenceTransformer")
    def setUp(self, mock_sentence_transformer, mock_weaviate):
        self.mock_model = MagicMock()
        mock_sentence_transformer.return_value = self.mock_model
        
        self.mock_client = MagicMock()
        mock_weaviate.return_value = self.mock_client
        
        self.evaluator = RAGEvaluator("https://s2umskuhqz23dftu3da19g.c0.asia-southeast1.gcp.weaviate.cloud", "uT1gw7iS9ncKF14lDqDmByDXeWLXwa1rI76p")
    
    def test_compute_semantic_similarity(self):
        self.mock_model.encode.return_value = [0.1, 0.2, 0.3]
        similarity_score = self.evaluator.compute_semantic_similarity("hello", "hi")
        self.assertIsInstance(similarity_score, float)
    
    def test_evaluate_response_bleu(self):
        scores = self.evaluator.evaluate_response("hello world", "hello")
        self.assertIn("BLEU", scores)
        self.assertGreaterEqual(scores["BLEU"], 0.0)
        self.assertLessEqual(scores["BLEU"], 1.0)
    
    def test_evaluate_response_rouge(self):
        scores = self.evaluator.evaluate_response("hello world", "hello")
        self.assertIn("ROUGE-1", scores)
        self.assertGreaterEqual(scores["ROUGE-1"], 0.0)
        self.assertLessEqual(scores["ROUGE-1"], 1.0)
    
    @patch("evaluate.ollama.chat")
    def test_run_evaluation(self, mock_chat):
        mock_chat.return_value = {"message": {"content": "Test response"}}
        self.evaluator.client.collections.get.return_value.query.near_vector.return_value.objects = [
            MagicMock(properties={"text": "Sample context"})
        ]
        with patch("builtins.open", unittest.mock.mock_open(read_data=json.dumps([
            {"question": "Test question", "answer": "Expected answer"}
        ]))):
            self.evaluator.run_evaluation()
        self.assertTrue(mock_chat.called)
    
    def test_empty_response(self):
        scores = self.evaluator.evaluate_response("hello world", "")
        self.assertAlmostEqual(scores["BLEU"], 0.0)
        self.assertAlmostEqual(scores["ROUGE-1"], 0.0)
        self.assertAlmostEqual(scores["ROUGE-L"], 0.0)
    
    def test_perfect_match(self):
        scores = self.evaluator.evaluate_response("hello", "hello")
        self.assertAlmostEqual(scores["ROUGE-1"], 1.0)
        self.assertAlmostEqual(scores["ROUGE-L"], 1.0)
    
    def test_partial_match(self):
        scores = self.evaluator.evaluate_response("hello world", "hello")
        self.assertLess(scores["BLEU"], 1.0)
        self.assertLess(scores["ROUGE-1"], 1.0)
        self.assertLess(scores["ROUGE-L"], 1.0)
    
if __name__ == "__main__":
    unittest.main()
