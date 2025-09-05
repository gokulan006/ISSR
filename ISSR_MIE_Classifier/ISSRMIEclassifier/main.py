#!/usr/bin/env python3
"""
MIE (Militarized Interstate Events) Classification System
Main Entry Point

This system uses the Enhanced MIE Classifier that combines:
- Traditional ML (TF-IDF + Naive Bayes/Random Forest)
- RAG (Retrieval Augmented Generation) with similar articles
- Ollama LLM classification with custom MIE Expert model
- Sentiment analysis and entity extraction
- Your MIC heuristics and death word analysis
"""

import sys
import os
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import the enhanced classifier
from ml.models.enhanced_mie_classifier import EnhancedMIEClassifier

def main():
    """Main entry point for the MIE classification system"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--quiet', action='store_true', help='Minimal console output')
    parser.add_argument('--json', action='store_true', help='Also print strict JSON coding block')
    parser.add_argument('--model', default='mie-expert', help='Ollama model name')
    parser.add_argument('--data', default='data/raw/final_data_true.csv', help='Training CSV path')
    args = parser.parse_args()

    if not args.quiet:
        print("Loading…")

    # Initialize enhanced classifier
    classifier = EnhancedMIEClassifier(model_name=args.model, verbose=not args.quiet)

    # Load and prepare data
    try:
        df = classifier.load_and_prepare_data(args.data)
        
        # Prepare features
        df['combined_text'] = df['Title'].fillna('') + ' ' + df['Subject '].fillna('') + ' ' + df['Text'].fillna('')
        X = df['combined_text']
        y = df['label']
        
        # Train enhanced model
        classifier.train_enhanced_model(X, y)
        
        # Create RAG embeddings for similar article retrieval
        classifier.create_rag_embeddings(df)
        
    except FileNotFoundError:
        print("Training data not found.")
        print("   Please ensure the file exists and try again.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    if not args.quiet:
        print("Ready. Enter article details (type 'quit' for title to exit).")
    
    try:
        # Lightweight interactive loop using existing method
        # If args.json set, show parsed JSON after verdict
        while True:
            title = input("Article Title (or 'quit'): ").strip()
            if title.lower() == 'quit':
                break
            subject = input("Subject: ").strip()
            text = input("Text: ").strip()
            if not any([title, subject, text]):
                continue
            result = classifier.predict_enhanced_mie(title, subject, text)
            print(f"Prediction: {'MIE' if result['final_prediction'] == 1 else 'NOT_MIE'}  (conf={result['confidence']:.2f})")
            if args.json:
                # When Ollama is available, classifier returns parsed Ollama JSON in result['ollama_analysis']? For strict JSON, print coding if present
                coding = result.get('ollama_analysis', {}).get('coding') if isinstance(result.get('ollama_analysis'), dict) else None
                if coding:
                    import json as _json
                    obj = {"classification": result['ollama_analysis'].get('classification','UNKNOWN'),
                           "reasoning": result['ollama_analysis'].get('reasoning',''),
                           "codeable": result.get('ollama_analysis',{}).get('codeable', True),
                           "coding": coding,
                           "countries_involved": result['ollama_analysis'].get('countries', []),
                           "missing_fields": result['ollama_analysis'].get('missing_fields', []),
                           "notes": result['ollama_analysis'].get('notes',''),
                           "confrontation_key": result['ollama_analysis'].get('confrontation_key','')}
                    print(_json.dumps(obj, ensure_ascii=False))
    except KeyboardInterrupt:
        if not args.quiet:
            print("\nGoodbye.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 