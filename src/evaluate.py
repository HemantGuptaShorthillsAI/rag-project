import json
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import weaviate
import ollama
from weaviate.classes.init import Auth
from sentence_transformers import util
import os
from dotenv import load_dotenv

load_dotenv()


class RAGEvaluator:
    def __init__(self, weaviate_url, weaviate_api_key, dataset_path="../assets/golden_dataset.json"):
        self.weaviate_url = weaviate_url
        self.weaviate_api_key = weaviate_api_key
        self.dataset_path = dataset_path
        self.model = self.load_model()
        self.client = self.init_weaviate_client()
    
    def load_model(self):
        """Load the sentence transformer model."""
        return SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
    
    def init_weaviate_client(self):
        """Initialize Weaviate client."""
        return weaviate.connect_to_weaviate_cloud(
            cluster_url=self.weaviate_url,
            auth_credentials=Auth.api_key(self.weaviate_api_key),
            additional_config=weaviate.config.AdditionalConfig(timeout=weaviate.config.Timeout(init=60)),
        )
    
    def compute_semantic_similarity(self, expected, generated):
        embedding1 = self.model.encode(expected, convert_to_tensor=True)
        embedding2 = self.model.encode(generated, convert_to_tensor=True)
        return util.pytorch_cos_sim(embedding1, embedding2).item()
    
    def evaluate_response(self, true_answer, chatbot_response):
        smooth = SmoothingFunction().method1
        bleu = sentence_bleu([true_answer.split()], chatbot_response.split(), smoothing_function=smooth)
        
        rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        rouge_scores = rouge.score(true_answer, chatbot_response)
        
        semantic_similarity = self.compute_semantic_similarity(true_answer, chatbot_response)
        
        return {
            "BLEU": bleu,
            "ROUGE-1": rouge_scores["rouge1"].fmeasure,
            "ROUGE-L": rouge_scores["rougeL"].fmeasure,
            "Semantic Similarity": semantic_similarity
        }
    
    def run_evaluation(self):
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            golden_data = json.load(f)
        
        results = []
        try:  
            for qa in golden_data:
                question, true_answer = qa["question"], qa["answer"]
                
                query_embedding = self.model.encode([question]).tolist()
                
                documents = self.client.collections.get("TextChunks")
                response = documents.query.near_vector(near_vector=query_embedding[0])
                
                retrieved_docs = [obj.properties["text"] for obj in response.objects]
                context = "\n".join(retrieved_docs)
                
                prompt = f"Context:\n{context}\n\nQuestion: {question}"
                chatbot_response = ollama.chat(model="llama3.2:3b", messages=[{"role": "user", "content": prompt}])["message"]["content"]
                
                print(f"Q: {question}")
                print(f"Chatbot Answer: {chatbot_response}")
                print(f"Expected Answer: {true_answer}")

                scores = self.evaluate_response(true_answer, chatbot_response)
                print(scores)
                
                results.append({
                    "question": question,
                    "expected_answer": true_answer,
                    "generated_answer": chatbot_response,
                    "BLEU": scores["BLEU"],
                    "ROUGE-1": scores["ROUGE-1"],
                    "ROUGE-L": scores["ROUGE-L"],
                    "Semantic Similarity": scores["Semantic Similarity"]
                })
                
                with open("../assets/evaluate.json", "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=4)
                    f.flush()
                    f.close()
                
                print("Evaluation completed! Results saved in evaluation_results.json")
        
        except KeyboardInterrupt:
            print("\nCtrl + C detected! Saving progress before exiting...")
            with open("../assets/evaluation_results.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)
            print("Final progress saved to evaluation_results.json. Exiting safely.")

# Example usage:
if __name__ == "__main__":
    WEAVIATE_URL = os.getenv("WEAVIATE_URL")
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
    evaluator = RAGEvaluator(WEAVIATE_URL,WEAVIATE_API_KEY)
    evaluator.run_evaluation()
