import pandas as pd
import requests
import json
import time
from typing import Dict, List, Tuple
import re

class MIEExpertTester:
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model_name = "mie-expert"
        
    def query_ollama(self, prompt: str) -> str:
        """Query the MIE Expert model via Ollama API"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            print(f"Error querying Ollama: {e}")
            return ""

    def format_article_for_classification(self, title: str, subject: str, text: str) -> str:
        """Format article data for MIE classification"""
        return f"""Analyze this article for MIE classification:

TITLE: {title}
SUBJECT: {subject}
TEXT: {text[:1000]}...

Is this an MIE? If yes, what type of action and what variables should be coded? Provide your analysis in a structured format."""

    def parse_mie_response(self, response: str) -> Dict:
        """Parse the MIE Expert response to extract classification and details"""
        result = {
            "classification": "UNKNOWN",
            "action_type": None,
            "reasoning": "",
            "coding_variables": {}
        }
        
        # Extract classification
        if "MIE (YES)" in response or "Classification: MIE (YES)" in response:
            result["classification"] = "MIE"
        elif "NOT_MIE" in response or "MIE (NO)" in response or "Classification: MIE (NO)" in response:
            result["classification"] = "NOT_MIE"
        
        # Extract action type
        action_match = re.search(r'Action Type:?\s*(\d+)', response, re.IGNORECASE)
        if action_match:
            result["action_type"] = int(action_match.group(1))
        
        # Extract reasoning
        reasoning_match = re.search(r'Reasoning:?\s*(.*?)(?=Action Type:|CODING|$)', response, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).strip()
        
        return result

    def test_sample_articles(self, df: pd.DataFrame, sample_size: int = 10) -> List[Dict]:
        """Test the model on a sample of articles"""
        results = []
        
        # Sample articles (mix of MIE and non-MIE)
        sample_df = df.sample(n=sample_size, random_state=42)
        
        print(f"Testing MIE Expert model on {sample_size} sample articles...")
        print("=" * 60)
        
        for idx, row in sample_df.iterrows():
            print(f"\nArticle {idx}: {row['Title'][:50]}...")
            print(f"Ground Truth: {'MIE' if row['Probable MIE'] == 1.0 else 'NOT MIE'}")
            
            # Format article for classification
            prompt = self.format_article_for_classification(
                row['Title'], 
                row['Subject '], 
                row['Text']
            )
            
            # Query the model
            print("Querying MIE Expert...")
            response = self.query_ollama(prompt)
            
            # Parse response
            parsed = self.parse_mie_response(response)
            
            # Compare with ground truth
            ground_truth = "MIE" if row['Probable MIE'] == 1.0 else "NOT_MIE"
            correct = parsed["classification"] == ground_truth
            
            result = {
                "article_id": idx,
                "title": row['Title'][:50] + "...",
                "ground_truth": ground_truth,
                "predicted": parsed["classification"],
                "correct": correct,
                "action_type": parsed["action_type"],
                "reasoning": parsed["reasoning"][:100] + "..." if len(parsed["reasoning"]) > 100 else parsed["reasoning"],
                "full_response": response[:200] + "..." if len(response) > 200 else response
            }
            
            results.append(result)
            
            print(f"Predicted: {parsed['classification']}")
            print(f"Correct: {'✓' if correct else '✗'}")
            print(f"Action Type: {parsed['action_type']}")
            
            # Add delay to avoid overwhelming the model
            time.sleep(1)
        
        return results

    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate accuracy metrics"""
        total = len(results)
        correct = sum(1 for r in results if r["correct"])
        accuracy = correct / total if total > 0 else 0
        
        # Breakdown by class
        mie_correct = sum(1 for r in results if r["ground_truth"] == "MIE" and r["correct"])
        mie_total = sum(1 for r in results if r["ground_truth"] == "MIE")
        mie_accuracy = mie_correct / mie_total if mie_total > 0 else 0
        
        not_mie_correct = sum(1 for r in results if r["ground_truth"] == "NOT_MIE" and r["correct"])
        not_mie_total = sum(1 for r in results if r["ground_truth"] == "NOT_MIE")
        not_mie_accuracy = not_mie_correct / not_mie_total if not_mie_total > 0 else 0
        
        return {
            "total_articles": total,
            "overall_accuracy": accuracy,
            "mie_accuracy": mie_accuracy,
            "not_mie_accuracy": not_mie_accuracy,
            "correct_predictions": correct,
            "incorrect_predictions": total - correct
        }

def main():
    print("MIE Expert Model Testing")
    print("=" * 40)
    
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv('final_data_true.csv')
    print(f"Loaded {len(df)} articles")
    
    # Check class distribution
    mie_count = (df['Probable MIE'] == 1.0).sum()
    not_mie_count = (df['Probable MIE'] == 0.0).sum()
    print(f"MIE articles: {mie_count}")
    print(f"Non-MIE articles: {not_mie_count}")
    
    # Initialize tester
    tester = MIEExpertTester()
    
    # Test on sample articles
    results = tester.test_sample_articles(df, sample_size=10)
    
    # Calculate metrics
    metrics = tester.calculate_metrics(results)
    
    # Print results
    print("\n" + "=" * 60)
    print("TESTING RESULTS")
    print("=" * 60)
    print(f"Total Articles Tested: {metrics['total_articles']}")
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.2%}")
    print(f"MIE Accuracy: {metrics['mie_accuracy']:.2%}")
    print(f"Non-MIE Accuracy: {metrics['not_mie_accuracy']:.2%}")
    print(f"Correct Predictions: {metrics['correct_predictions']}")
    print(f"Incorrect Predictions: {metrics['incorrect_predictions']}")
    
    # Print detailed results
    print("\n" + "=" * 60)
    print("DETAILED RESULTS")
    print("=" * 60)
    for result in results:
        print(f"\nArticle: {result['title']}")
        print(f"Ground Truth: {result['ground_truth']}")
        print(f"Predicted: {result['predicted']}")
        print(f"Correct: {'✓' if result['correct'] else '✗'}")
        if result['action_type']:
            print(f"Action Type: {result['action_type']}")
        print(f"Reasoning: {result['reasoning']}")

if __name__ == "__main__":
    main() 