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
import csv
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import the enhanced classifier
from ml.models.enhanced_mie_classifier import EnhancedMIEClassifier
from utils.csv_exporter import MIEExporter
from utils.file_parser import FileParser
from utils.visualizer import MIEVisualizer

def main():
    """Main entry point for the MIE classification system"""
    parser = argparse.ArgumentParser(description='Enhanced MIE Classification System')
    parser.add_argument('--quiet', action='store_true', help='Minimal console output')
    parser.add_argument('--json', action='store_true', help='Also print strict JSON coding block')
    parser.add_argument('--model', default='mie-expert', help='Ollama model name')
    parser.add_argument('--data', default='data/raw/final_data_true.csv', help='Training CSV path')
    parser.add_argument('--input-file', help='Input text file for batch processing')
    parser.add_argument('--output-csv', help='Output CSV file for results')
    parser.add_argument('--bulk-test', type=int, help='Run bulk test on N random articles')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization of results')
    parser.add_argument('--enhanced', action='store_true', help='Show enhanced interactive mode with detailed results')
    args = parser.parse_args()

    if not args.quiet:
        print("Loading…")

    # Initialize enhanced classifier and utilities
    classifier = EnhancedMIEClassifier(model_name=args.model, verbose=not args.quiet)
    exporter = MIEExporter() if args.output_csv else None
    file_parser = FileParser() if args.input_file else None
    visualizer = MIEVisualizer() if args.visualize else None

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
    
    # Process file input if provided
    if args.input_file:
        try:
            if not args.quiet:
                print(f"📁 Processing file: {args.input_file}")
            
            # Parse input file
            articles = file_parser.parse_file(args.input_file)
            articles = file_parser.validate_articles(articles)
            
            if not args.quiet:
                print(f"📄 Found {len(articles)} articles to process")
            
            # Process each article
            results = []
            for i, article in enumerate(articles, 1):
                if not args.quiet:
                    print(f"🔍 Processing article {i}/{len(articles)}: {article['title'][:50]}...")
                
                result = classifier.predict_enhanced_mie(
                    article['title'], 
                    article['subject'], 
                    article['text']
                )
                
                # Add article metadata
                result['article_id'] = f"file_{i}"
                result['title'] = article['title']
                result['subject'] = article['subject']
                result['text'] = article['text']
                
                results.append(result)
                
                # Show result
                print(f"  Result: {'MIE' if result['final_prediction'] == 1 else 'NOT_MIE'} (conf={result['confidence']:.2f})")
                
                # Export to CSV if requested
                if exporter and args.output_csv:
                    exporter.export_single_result(result, args.output_csv, 
                                                article['title'], article['subject'], article['text'])
            
            # Export batch results if CSV output requested
            if exporter and args.output_csv:
                exporter.export_batch_results(results, args.output_csv.replace('.csv', '_batch.csv'))
                
                # Generate summary report
                summary_path = args.output_csv.replace('.csv', '_summary.txt')
                exporter.create_summary_report(args.output_csv, summary_path)
            
            # Create visualization if requested
            if visualizer and args.visualize:
                html_path = args.output_csv.replace('.csv', '_report.html') if args.output_csv else None
                visualizer.create_html_report(results, html_path)
            
            if not args.quiet:
                print(f"✅ File processing complete! Processed {len(articles)} articles.")
            
            return
            
        except Exception as e:
            print(f"❌ Error processing file: {e}")
            return
    
    if not args.quiet:
        print("Ready. Enter article details (type 'quit' for title to exit).")
    
    try:
        # Choose interactive mode based on arguments
        if args.enhanced:
            # Use enhanced interactive mode with full details
            classifier.interactive_mode()
        else:
            # Lightweight interactive loop
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
                
                # Export to CSV if requested
                if exporter and args.output_csv:
                    exporter.export_single_result(result, args.output_csv, title, subject, text)
                
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