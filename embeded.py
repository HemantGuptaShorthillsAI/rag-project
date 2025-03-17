import os
import weaviate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from weaviate.classes.init import Auth
from sentence_transformers import SentenceTransformer

class WeaviateTextEmbedder:
    def __init__(self, folder_path, weaviate_url, weaviate_api_key):
        self.folder_path = folder_path
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=Auth.api_key(weaviate_api_key),
        )
        self.model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

    def load_text_files(self):
        text_data = []
        for file_name in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, file_name)
            if os.path.isfile(file_path) and file_name.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as file:
                    text_data.append(file.read())
        return text_data

    def split_text(self, text):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        return text_splitter.split_text(text)

    def setup_weaviate(self):
        existing_collections = self.client.collections.list_all()
        if "TextChunks" not in existing_collections:
            self.client.collections.create(
                name="TextChunks",
                vectorizer_config=weaviate.Vectorizer.NONE
            )

    def insert_data(self, text_chunks):
        collection = self.client.collections.get("TextChunks")
        with collection.batch.dynamic() as batch:
            for chunk in text_chunks:
                vector = self.model.encode(chunk)
                batch.add_object(
                    properties={"text": chunk},
                    vector=vector
                )

    def run(self):
        self.setup_weaviate()
        all_texts = self.load_text_files()
        all_chunks = [chunk for text in all_texts for chunk in self.split_text(text)]
        self.insert_data(all_chunks)
        print("Embedding and storage in Weaviate completed.")
        self.client.close()

if __name__ == "__main__":
    folder_path = "unicorn_startups_text"  # Folder containing text files
    WEAVIATE_URL="https://wm3dtb9mqooxlzl0yp0ag.c0.asia-southeast1.gcp.weaviate.cloud"
    WEAVIATE_API_KEY="jBm4pMt7E4Mh4PHGdH3zShw0PRmoHWPB4nl8"

    
    embedder = WeaviateTextEmbedder(folder_path, WEAVIATE_URL, WEAVIATE_API_KEY)
    embedder.run()
