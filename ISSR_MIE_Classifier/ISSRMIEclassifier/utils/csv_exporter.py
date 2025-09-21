"""
CSV Export functionality for MIE Classifier results
Provides structured, readable output for analysis and sharing
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

class MIEExporter:
    """Export MIE classification results to CSV"""
    
    def __init__(self):
        self.fieldnames = [
            'timestamp',
            'article_id',
            'title',
            'subject', 
            'text_preview',
            'final_prediction',
            'confidence',
            'ml_prediction',
            'ollama_classification',
            'ollama_reasoning',
            'countries_involved',
            'total_fatalities',
            'sentiment_score',
            'mie_word_count',
            'death_word_count',
            'similar_articles_count',
            'rag_context',
            'full_json_response'
        ]
    
    def export_single_result(self, result: Dict, output_path: str, 
                           title: str = "", subject: str = "", text: str = ""):
        """
        Export a single classification result to CSV
        
        Args:
            result: Classification result dictionary
            output_path: Output CSV file path
            title: Article title
            subject: Article subject
            text: Article text
        """
        output_file = Path(output_path)
        file_exists = output_file.exists()
        
        # Prepare row data
        row_data = self._prepare_row_data(result, title, subject, text)
        
        with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            
            # Write header if new file
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(row_data)
        
        print(f"📄 Results exported to: {output_path}")
    
    def export_batch_results(self, results: List[Dict], output_path: str):
        """
        Export multiple classification results to CSV
        
        Args:
            results: List of result dictionaries
            output_path: Output CSV file path
        """
        output_file = Path(output_path)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()
            
            for i, result in enumerate(results):
                row_data = self._prepare_row_data(result, 
                                                result.get('title', f'Article_{i}'),
                                                result.get('subject', ''),
                                                result.get('text', ''))
                writer.writerow(row_data)
        
        print(f"📊 Batch results exported: {len(results)} articles to {output_path}")
    
    def _prepare_row_data(self, result: Dict, title: str, subject: str, text: str) -> Dict:
        """Prepare a single row of data for CSV export"""
        
        # Get Ollama analysis data safely
        ollama_data = result.get('ollama_analysis', {})
        
        # Get entity data safely
        entities = result.get('entities', {})
        
        # Get sentiment data safely
        sentiment = result.get('sentiment_analysis', {})
        
        # Get similar articles safely
        similar_articles = result.get('similar_articles', [])
        
        # Prepare RAG context
        rag_context = ""
        if similar_articles:
            rag_context = "; ".join([
                f"{art.get('title', 'Unknown')[:30]}... (sim: {art.get('similarity', 0):.2f})"
                for art in similar_articles[:3]
            ])
        
        return {
            'timestamp': datetime.now().isoformat(),
            'article_id': result.get('article_id', ''),
            'title': title[:100] if title else '',
            'subject': subject[:100] if subject else '',
            'text_preview': text[:200] + "..." if len(text) > 200 else text,
            'final_prediction': 'MIE' if result.get('final_prediction') == 1 else 'NOT_MIE',
            'confidence': round(result.get('confidence', 0), 3),
            'ml_prediction': 'MIE' if result.get('ml_prediction') == 1 else 'NOT_MIE',
            'ollama_classification': ollama_data.get('classification', 'UNKNOWN'),
            'ollama_reasoning': ollama_data.get('reasoning', '')[:200] + "..." if len(ollama_data.get('reasoning', '')) > 200 else ollama_data.get('reasoning', ''),
            'countries_involved': ', '.join(entities.get('countries', [])),
            'total_fatalities': entities.get('total_fatalities', 0),
            'sentiment_score': round(sentiment.get('sentiment_compound', 0), 3),
            'mie_word_count': sentiment.get('mie_word_count', 0),
            'death_word_count': sentiment.get('death_word_count', 0),
            'similar_articles_count': len(similar_articles),
            'rag_context': rag_context,
            'full_json_response': json.dumps(result, ensure_ascii=False, default=str)
        }
    
    def create_summary_report(self, csv_path: str, output_path: str = None):
        """
        Create a summary report from CSV results
        
        Args:
            csv_path: Path to CSV file with results
            output_path: Optional output path for summary
        """
        if not output_path:
            output_path = str(Path(csv_path).with_suffix('_summary.txt'))
        
        # Read CSV and generate summary
        results = []
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            results = list(reader)
        
        # Generate summary
        summary = self._generate_summary(results)
        
        # Write summary
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"📋 Summary report created: {output_path}")
        return summary
    
    def _generate_summary(self, results: List[Dict]) -> str:
        """Generate summary statistics from results"""
        total = len(results)
        mie_count = sum(1 for r in results if r['final_prediction'] == 'MIE')
        not_mie_count = total - mie_count
        
        avg_confidence = sum(float(r['confidence']) for r in results) / total if total > 0 else 0
        avg_sentiment = sum(float(r['sentiment_score']) for r in results) / total if total > 0 else 0
        
        # Country analysis
        all_countries = []
        for r in results:
            countries = r['countries_involved'].split(', ') if r['countries_involved'] else []
            all_countries.extend(countries)
        
        country_counts = {}
        for country in all_countries:
            if country:
                country_counts[country] = country_counts.get(country, 0) + 1
        
        top_countries = sorted(country_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        summary = f"""
MIE CLASSIFICATION SUMMARY REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 OVERALL STATISTICS
Total Articles Analyzed: {total}
MIE Predictions: {mie_count} ({mie_count/total*100:.1f}%)
NOT_MIE Predictions: {not_mie_count} ({not_mie_count/total*100:.1f}%)

📈 PERFORMANCE METRICS
Average Confidence: {avg_confidence:.3f}
Average Sentiment Score: {avg_sentiment:.3f}

🌍 COUNTRY ANALYSIS
Top 5 Countries Involved:
"""
        for country, count in top_countries:
            summary += f"  {country}: {count} occurrences\n"
        
        summary += f"""
📋 DETAILED BREAKDOWN
"""
        
        # Add breakdown by prediction type
        for prediction_type in ['MIE', 'NOT_MIE']:
            type_results = [r for r in results if r['final_prediction'] == prediction_type]
            if type_results:
                avg_conf = sum(float(r['confidence']) for r in type_results) / len(type_results)
                summary += f"{prediction_type} Articles: {len(type_results)} (avg confidence: {avg_conf:.3f})\n"
        
        return summary
