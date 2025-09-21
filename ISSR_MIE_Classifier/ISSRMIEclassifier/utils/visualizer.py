"""
Visualization utilities for MIE Classifier results
Creates charts and HTML reports for presentation
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

class MIEVisualizer:
    """Create visualizations for MIE classification results"""
    
    def __init__(self):
        self.output_dir = Path("visualizations")
        self.output_dir.mkdir(exist_ok=True)
    
    def create_html_report(self, results: List[Dict], output_path: str = None):
        """
        Create HTML report with visualizations
        
        Args:
            results: List of classification results
            output_path: Output HTML file path
        """
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"mie_report_{timestamp}.html"
        
        # Generate HTML content
        html_content = self._generate_html_report(results)
        
        # Write HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📊 HTML report created: {output_path}")
        return output_path
    
    def _generate_html_report(self, results: List[Dict]) -> str:
        """Generate HTML report content"""
        
        # Calculate statistics
        total = len(results)
        mie_count = sum(1 for r in results if r.get('final_prediction') == 1)
        not_mie_count = total - mie_count
        
        avg_confidence = sum(r.get('confidence', 0) for r in results) / total if total > 0 else 0
        
        # Country analysis
        all_countries = []
        for r in results:
            countries = r.get('entities', {}).get('countries', [])
            all_countries.extend(countries)
        
        country_counts = {}
        for country in all_countries:
            country_counts[country] = country_counts.get(country, 0) + 1
        
        top_countries = sorted(country_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Sentiment analysis
        sentiment_scores = [r.get('sentiment_analysis', {}).get('sentiment_compound', 0) for r in results]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MIE Classification Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            border-bottom: 3px solid #2c3e50;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #2c3e50;
            margin: 0;
        }}
        .header p {{
            color: #7f8c8d;
            margin: 10px 0 0 0;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .stat-card h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            opacity: 0.9;
        }}
        .stat-card .number {{
            font-size: 2.5em;
            font-weight: bold;
            margin: 0;
        }}
        .chart-container {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .chart-title {{
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 15px;
        }}
        .bar-chart {{
            display: flex;
            align-items: end;
            height: 200px;
            gap: 10px;
            padding: 20px 0;
        }}
        .bar {{
            flex: 1;
            background: linear-gradient(to top, #3498db, #2980b9);
            border-radius: 5px 5px 0 0;
            display: flex;
            align-items: end;
            justify-content: center;
            color: white;
            font-weight: bold;
            min-height: 30px;
        }}
        .bar.mie {{ background: linear-gradient(to top, #e74c3c, #c0392b); }}
        .bar.not-mie {{ background: linear-gradient(to top, #27ae60, #229954); }}
        .bar-label {{
            text-align: center;
            margin-top: 10px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .countries-list {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px;
        }}
        .country-item {{
            background: #ecf0f1;
            padding: 10px;
            border-radius: 5px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .country-name {{
            font-weight: bold;
            color: #2c3e50;
        }}
        .country-count {{
            background: #3498db;
            color: white;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 12px;
        }}
        .results-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        .results-table th,
        .results-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .results-table th {{
            background-color: #34495e;
            color: white;
        }}
        .results-table tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .prediction.mie {{ color: #e74c3c; font-weight: bold; }}
        .prediction.not-mie {{ color: #27ae60; font-weight: bold; }}
        .confidence {{
            background: #f39c12;
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 MIE Classification Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Articles</h3>
                <div class="number">{total}</div>
            </div>
            <div class="stat-card">
                <h3>MIE Predictions</h3>
                <div class="number">{mie_count}</div>
            </div>
            <div class="stat-card">
                <h3>NOT_MIE Predictions</h3>
                <div class="number">{not_mie_count}</div>
            </div>
            <div class="stat-card">
                <h3>Avg Confidence</h3>
                <div class="number">{avg_confidence:.2f}</div>
            </div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">📊 Classification Distribution</div>
            <div class="bar-chart">
                <div style="flex: 1;">
                    <div class="bar mie" style="height: {(mie_count/total*150) if total > 0 else 0}px;">
                        {mie_count}
                    </div>
                    <div class="bar-label">MIE ({mie_count/total*100:.1f}%)</div>
                </div>
                <div style="flex: 1;">
                    <div class="bar not-mie" style="height: {(not_mie_count/total*150) if total > 0 else 0}px;">
                        {not_mie_count}
                    </div>
                    <div class="bar-label">NOT_MIE ({not_mie_count/total*100:.1f}%)</div>
                </div>
            </div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">🌍 Top Countries Involved</div>
            <div class="countries-list">
        """
        
        for country, count in top_countries:
            html += f"""
                <div class="country-item">
                    <span class="country-name">{country}</span>
                    <span class="country-count">{count}</span>
                </div>
            """
        
        html += """
            </div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">📋 Detailed Results</div>
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Article</th>
                        <th>Prediction</th>
                        <th>Confidence</th>
                        <th>Countries</th>
                        <th>Sentiment</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        # Add table rows for each result
        for i, result in enumerate(results[:20]):  # Limit to first 20 for readability
            title = result.get('title', f'Article {i+1}')[:50]
            prediction = 'MIE' if result.get('final_prediction') == 1 else 'NOT_MIE'
            confidence = result.get('confidence', 0)
            countries = ', '.join(result.get('entities', {}).get('countries', [])[:3])
            sentiment = result.get('sentiment_analysis', {}).get('sentiment_compound', 0)
            
            html += f"""
                    <tr>
                        <td>{title}...</td>
                        <td><span class="prediction {prediction.lower().replace('_', '-')}">{prediction}</span></td>
                        <td><span class="confidence">{confidence:.2f}</span></td>
                        <td>{countries}</td>
                        <td>{sentiment:.3f}</td>
                    </tr>
            """
        
        html += """
                </tbody>
            </table>
        </div>
        
        <div style="text-align: center; margin-top: 30px; color: #7f8c8d;">
            <p>Report generated by Enhanced MIE Classification System</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html
    
    def create_visualization_from_csv(self, csv_path: str, output_path: str = None):
        """
        Create visualization from CSV results file
        
        Args:
            csv_path: Path to CSV file with results
            output_path: Output HTML file path
        """
        try:
            # Read CSV file
            df = pd.read_csv(csv_path)
            
            # Convert to results format
            results = []
            for _, row in df.iterrows():
                result = {
                    'final_prediction': 1 if row['final_prediction'] == 'MIE' else 0,
                    'confidence': row['confidence'],
                    'title': row['title'],
                    'entities': {
                        'countries': row['countries_involved'].split(', ') if pd.notna(row['countries_involved']) else []
                    },
                    'sentiment_analysis': {
                        'sentiment_compound': row['sentiment_score']
                    }
                }
                results.append(result)
            
            # Create HTML report
            return self.create_html_report(results, output_path)
            
        except Exception as e:
            print(f"❌ Error creating visualization from CSV: {e}")
            return None
