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

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import the enhanced classifier
from ml.models.enhanced_mie_classifier import EnhancedMIEClassifier

def main():
    """Main entry point for the MIE classification system"""
    
    print("🚀 ENHANCED MIE CLASSIFICATION SYSTEM")
    print("Combines: ML + RAG + Ollama + MIC Heuristics")
    print("=" * 50)
    
    # Initialize enhanced classifier with correct model name
    classifier = EnhancedMIEClassifier(model_name="mie-expert")
    
    # Check Ollama connection
    if classifier.check_ollama():
        print("✅ Ollama connected! Using full system (ML + RAG + LLM)")
    else:
        print("❌ Ollama not available. Using ML + RAG only.")
        print("   To enable LLM: run 'ollama serve' and './setup_ollama.sh'")
    
    # Load and prepare data
    print("\n📊 Loading training data...")
    try:
        df = classifier.load_and_prepare_data('data/raw/final_data_true.csv')
        print(f"✅ Loaded {len(df)} articles for training")
        
        # Prepare features
        df['combined_text'] = df['Title'].fillna('') + ' ' + df['Subject '].fillna('') + ' ' + df['Text'].fillna('')
        X = df['combined_text']
        y = df['label']
        
        # Train enhanced model
        print("🤖 Training enhanced model...")
        classifier.train_enhanced_model(X, y)
        print("✅ Model trained successfully!")
        
        # Create RAG embeddings for similar article retrieval
        print("🔍 Creating RAG embeddings...")
        classifier.create_rag_embeddings(df)
        print("✅ RAG system ready!")
        
    except FileNotFoundError:
        print("⚠️  Training data not found at 'data/raw/final_data_true.csv'")
        print("   Please ensure the file exists and try again.")
        return
    except Exception as e:
        print(f"⚠️  Error loading data: {e}")
        print("   Please check your data files and try again.")
        return
    
    # Start interactive mode
    print("\n🎯 Starting Interactive Classification Mode")
    print("Enter article details to classify as MIE or NOT_MIE")
    print("Type 'quit' for title to exit")
    print("-" * 50)
    
    try:
        classifier.interactive_mode()
    except KeyboardInterrupt:
        print("\n\n👋 Thanks for using the Enhanced MIE Classifier!")
    except Exception as e:
        print(f"\n❌ Error in interactive mode: {e}")
        print("Please restart the program.")

if __name__ == "__main__":
    main() 