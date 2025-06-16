import json
import os
import random
from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from transformers import pipeline
import ollama
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import bert_score
import csv
import pandas as pd
import google.generativeai as genai
import time
load_dotenv()


class VectorDB:
    def __init__(self, weaviate_url, weaviate_api_key):
        self.weaviate_url = weaviate_url
        self.weaviate_api_key = weaviate_api_key
        self.client = self.init_weaviate_client()
    
    def init_weaviate_client(self):
        return weaviate.connect_to_weaviate_cloud(
            cluster_url=self.weaviate_url,
            auth_credentials=Auth.api_key(self.weaviate_api_key),
        )
    
    def retrieve_context(self, query_embedding):
        documents = self.client.collections.get("TextChunks")
        response = documents.query.near_vector(near_vector=query_embedding[0])
        return [obj.properties["text"] for obj in response.objects]


class LLMModel:
    def __init__(self, model_name="gemini-2.0-flash"):
        self.model_name = model_name
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(self.model_name)

    def generate_response(self, prompt, context=""):
        prompt_parts = [
            f"""You are an expert fact-checking AI. Answer the question **strictly using the context**. \n            - If the context does not contain the answer, reply with **\"I don't know.\"**\n            - Ensure answers match expected numerical values, names, and key facts.\n            - Give to the point and concise answer no useless explanations and don't extend answers just give dates and figures where asked don't complete sentences.\n
        ### Context:\n        {context}\n
        ### Question:\n        {prompt}"""
        ]
        try:
            response = self.model.generate_content(prompt_parts, generation_config=genai.GenerationConfig(temperature=0.1))
            return response.text
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I don't know."


class RAGEvaluator:
    def __init__(self, weaviate_url, weaviate_api_key, dataset_path="../assets/golden_dataset.json", model_name="llama3.2:3b"):
        self.dataset_path = dataset_path
        self.embedder = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
        self.vector_db = VectorDB(weaviate_url, weaviate_api_key)
        self.llm = LLMModel(model_name)
        self.model_name1 = "roberta-large-mnli"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name1)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name1)
        self.model.eval()
    
    def compute_semantic_similarity(self, expected, generated):
        embedding1 = self.embedder.encode(expected, convert_to_tensor=True)
        embedding2 = self.embedder.encode(generated, convert_to_tensor=True)
        return util.pytorch_cos_sim(embedding1, embedding2).item()
    
    def compute_nli(self, premise, hypothesis):
        inputs = self.tokenizer.encode_plus(premise, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze()
        labels = ['contradiction', 'neutral', 'entailment']
        pred_label = labels[torch.argmax(probs).item()]
        return pred_label, probs.tolist()
    
    def evaluate_response(self, expected, generated):
        smooth = SmoothingFunction().method1
        bleu = sentence_bleu([expected.split()], generated.split(), smoothing_function=smooth)
        rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        rouge_scores = rouge.score(expected, generated)
        semantic_similarity = self.compute_semantic_similarity(expected, generated)
        label, prob = self.compute_nli(expected, generated)
        meteor = meteor_score([expected.split()], generated.split())
        P, R, F1 = bert_score.score([generated], [expected], lang="en", rescale_with_baseline=True)
        bert_f1 = F1.item()
        expected_tokens = set(expected.lower().split())
        generated_tokens = set(generated.lower().split())
        common_tokens = expected_tokens & generated_tokens
        precision = len(common_tokens) / len(generated_tokens) if generated_tokens else 0
        recall = len(common_tokens) / len(expected_tokens) if expected_tokens else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
        
        return {
            "BLEU": bleu,
            "ROUGE-1": rouge_scores["rouge1"].fmeasure,
            "ROUGE-L": rouge_scores["rougeL"].fmeasure,
            "Semantic Similarity": semantic_similarity,
            "NLI_RESULT": label,
            "CONFIDENCE": prob,
            "METEOR": round(meteor, 4),
            "BERT_F1": round(bert_f1, 4),
            "F1_SCORE": round(f1, 4)
        }
    
    def run_evaluation(self):
        with open(self.dataset_path, "r", encoding="utf-8-sig") as f:
            golden_data = json.load(f)

        # num_samples = min(500, len(golden_data))
        # sampled_questions = random.sample(golden_data, num_samples)

        results = []
        csv_columns = [
            "question", "expected_answer", "generated_answer", "BLEU", "ROUGE-1", "ROUGE-L",
            "Semantic Similarity", "NLI_RESULT", "Contradiction", "Neutral", "Entailment", "METEOR", "BERT_F1", "F1_SCORE"
        ]
        
        try:
            i=1
            for qa in golden_data:
                if i==15:
                    i=1
                    time.sleep(60)
                question, expected_answer = qa["question"], qa["answer"]
                query_embedding = self.embedder.encode([question]).tolist()
                retrieved_docs = self.vector_db.retrieve_context(query_embedding)
                context = "\n".join(retrieved_docs)
                
                prompt = f"Context:\n{context}\n\nQuestion: {question}\n\n Use the above pieces of retrieved context to answer the question.\n answer concise and to the point. \n If you don't know the answer, say that you don't know."
                generated_answer = self.llm.generate_response(prompt)
                
                print(f"Q: {question}\nChatbot Answer: {generated_answer}\nExpected Answer: {expected_answer}")
                scores = self.evaluate_response(expected_answer, generated_answer)
                print(scores)
                
                contradiction, neutral, entailment = scores["CONFIDENCE"]
                result_entry={
                    "question": question,
                    "expected_answer": expected_answer,
                    "generated_answer": generated_answer,
                    "BLEU": scores["BLEU"],
                    "ROUGE-1": scores["ROUGE-1"],
                    "ROUGE-L": scores["ROUGE-L"],
                    "Semantic Similarity": scores["Semantic Similarity"],
                    "NLI_RESULT": scores["NLI_RESULT"],
                    "Contradiction": contradiction,
                    "Neutral": neutral,
                    "Entailment": entailment,
                    "METEOR": scores["METEOR"],
                    "BERT_F1": scores["BERT_F1"],
                    "F1_SCORE": scores["F1_SCORE"]
                }
                results.append(result_entry)
                i+=1
                df = pd.DataFrame(results, columns=csv_columns)
                df.to_csv("../assets/ev.csv", index=False, encoding='utf-8-sig',quoting=csv.QUOTE_ALL)
                
                print("Evaluation completed! Results saved in evaluation.csv")
        
        except KeyboardInterrupt:
            print("\nCtrl + C detected! Saving progress before exiting...")
            df = pd.DataFrame(results,columns=csv_columns)
            df.to_csv("../assets/ev.csv", index=False, encoding='utf-8-sig',quoting=csv.QUOTE_ALL)
            print("Final progress saved. Exiting safely.")


if __name__ == "__main__":
    WEAVIATE_URL = os.getenv("WEAVIATE_URL")
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
    evaluator = RAGEvaluator(WEAVIATE_URL, WEAVIATE_API_KEY, model_name="gemini-2.0-flash")
    evaluator.run_evaluation()
