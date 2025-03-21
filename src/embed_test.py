import unittest
from unittest.mock import patch, MagicMock, mock_open
from weaviate.classes.init import Auth
from sentence_transformers import SentenceTransformer
from embeded import WeaviateTextEmbedder  # Replace 'embeded' with the actual module name

class TestWeaviateTextEmbedder(unittest.TestCase):
    
    @patch("embeded.weaviate.connect_to_weaviate_cloud")
    @patch("embeded.SentenceTransformer")
    def setUp(self, mock_sentence_transformer, mock_weaviate):
        self.mock_client = MagicMock()
        mock_weaviate.return_value = self.mock_client
        
        self.mock_model = MagicMock()
        mock_sentence_transformer.return_value = self.mock_model
        
        self.embedder = WeaviateTextEmbedder("test_folder", "https://s2umskuhqz23dftu3da19g.c0.asia-southeast1.gcp.weaviate.cloud", "tesuT1gw7iS9ncKF14lDqDmByDXeWLXwa1rI76pt_key")
    
    @patch("os.path.isfile", return_value=True)
    @patch("os.listdir", return_value=["file1.txt", "file2.txt"])
    @patch("builtins.open", new_callable=mock_open)
    def test_load_text_files(self, mock_file, mock_listdir, mock_isfile):
        mock_file.side_effect = [
            mock_open(read_data="Text1").return_value,
            mock_open(read_data="Text2").return_value
        ]

        result = self.embedder.load_text_files()
        self.assertEqual(result, ["Text1", "Text2"])


    
    def test_split_text(self):
        long_text = "This is a long text. " * 50  # Ensure multiple chunks
        chunks = self.embedder.split_text(long_text)
        self.assertGreater(len(chunks), 1)  # Should split into multiple chunks

    def test_setup_weaviate(self):
        self.mock_client.collections.list_all.return_value = []  # No existing collections
        self.embedder.setup_weaviate()
        self.mock_client.collections.create.assert_called_once()
    
    @patch("embeded.SentenceTransformer.encode")
    def test_insert_data(self, mock_encode):
        mock_encode.return_value = [0.1, 0.2, 0.3]  # Dummy vector
        mock_collection = MagicMock()
        self.mock_client.collections.get.return_value = mock_collection
        
        self.embedder.insert_data(["Sample chunk"])
        mock_collection.batch.dynamic.assert_called()
    
    @patch("embeded.WeaviateTextEmbedder.load_text_files")
    @patch("embeded.WeaviateTextEmbedder.split_text")
    @patch("embeded.WeaviateTextEmbedder.insert_data")
    def test_run(self, mock_insert_data, mock_split_text, mock_load_text_files):
        mock_load_text_files.return_value = ["Text1"]
        mock_split_text.return_value = ["Chunk1"]
        
        self.embedder.run()
        mock_insert_data.assert_called_with(["Chunk1"])
    
    def test_empty_text_files(self):
        with patch("embeded.WeaviateTextEmbedder.load_text_files", return_value=[]):
            self.assertEqual(self.embedder.load_text_files(), [])
    
    def test_empty_split_text(self):
        self.assertEqual(self.embedder.split_text(""), [])
    
if __name__ == "__main__":
    unittest.main()
