"""
File input parsing utilities for MIE Classifier
Supports multiple formats: .txt, .csv, .json for batch processing
"""

import csv
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

class FileParser:
    """Parse input files for MIE classification"""
    
    def __init__(self):
        self.supported_formats = ['.txt', '.csv', '.json']
    
    def parse_file(self, file_path: str) -> List[Dict[str, str]]:
        """
        Parse input file and return list of articles
        
        Args:
            file_path: Path to input file
            
        Returns:
            List of dictionaries with 'title', 'subject', 'text' keys
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix not in self.supported_formats:
            raise ValueError(f"Unsupported format: {file_path.suffix}. Supported: {self.supported_formats}")
        
        if file_path.suffix == '.txt':
            return self._parse_txt(file_path)
        elif file_path.suffix == '.csv':
            return self._parse_csv(file_path)
        elif file_path.suffix == '.json':
            return self._parse_json(file_path)
    
    def _parse_txt(self, file_path: Path) -> List[Dict[str, str]]:
        """Parse plain text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Split by double newlines or common separators
        articles = []
        sections = content.split('\n\n')
        
        for i, section in enumerate(sections):
            if section.strip():
                # Try to extract title from first line
                lines = section.strip().split('\n')
                title = lines[0][:100] if lines else f"Article_{i+1}"
                
                # Rest is text content
                text = '\n'.join(lines[1:]) if len(lines) > 1 else section.strip()
                
                articles.append({
                    'title': title,
                    'subject': '',
                    'text': text
                })
        
        return articles
    
    def _parse_csv(self, file_path: Path) -> List[Dict[str, str]]:
        """Parse CSV file with article data"""
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")
        
        articles = []
        
        # Try to detect column names (flexible)
        title_col = self._find_column(df, ['title', 'Title', 'TITLE', 'article_title'])
        subject_col = self._find_column(df, ['subject', 'Subject', 'SUBJECT', 'summary'])
        text_col = self._find_column(df, ['text', 'Text', 'TEXT', 'content', 'body'])
        
        for idx, row in df.iterrows():
            title = str(row[title_col]) if title_col and pd.notna(row[title_col]) else f"Article_{idx+1}"
            subject = str(row[subject_col]) if subject_col and pd.notna(row[subject_col]) else ""
            text = str(row[text_col]) if text_col and pd.notna(row[text_col]) else ""
            
            articles.append({
                'title': title,
                'subject': subject,
                'text': text
            })
        
        return articles
    
    def _parse_json(self, file_path: Path) -> List[Dict[str, str]]:
        """Parse JSON file with article data"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        articles = []
        
        # Handle different JSON structures
        if isinstance(data, list):
            # List of articles
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    articles.append(self._extract_article_from_dict(item, i))
                else:
                    articles.append({
                        'title': f"Article_{i+1}",
                        'subject': '',
                        'text': str(item)
                    })
        elif isinstance(data, dict):
            # Single article or structured data
            if 'articles' in data:
                # Multiple articles in 'articles' key
                for i, item in enumerate(data['articles']):
                    articles.append(self._extract_article_from_dict(item, i))
            else:
                # Single article
                articles.append(self._extract_article_from_dict(data, 0))
        
        return articles
    
    def _extract_article_from_dict(self, item: Dict, index: int) -> Dict[str, str]:
        """Extract article data from dictionary"""
        title = item.get('title', item.get('Title', f"Article_{index+1}"))
        subject = item.get('subject', item.get('Subject', ''))
        text = item.get('text', item.get('Text', item.get('content', '')))
        
        return {
            'title': str(title) if title else f"Article_{index+1}",
            'subject': str(subject) if subject else '',
            'text': str(text) if text else ''
        }
    
    def _find_column(self, df: pd.DataFrame, possible_names: List[str]) -> str:
        """Find column name from list of possible names"""
        for name in possible_names:
            if name in df.columns:
                return name
        return None
    
    def validate_articles(self, articles: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Validate and clean articles
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            Validated and cleaned articles
        """
        valid_articles = []
        
        for i, article in enumerate(articles):
            # Ensure required fields exist
            if not article.get('title'):
                article['title'] = f"Article_{i+1}"
            
            if not article.get('text'):
                print(f"⚠️  Warning: Article {i+1} has no text content, skipping")
                continue
            
            # Truncate long fields
            article['title'] = article['title'][:200]
            article['subject'] = article['subject'][:200] if article.get('subject') else ''
            article['text'] = article['text'][:5000]  # Limit text length
            
            valid_articles.append(article)
        
        return valid_articles
    
    def create_sample_files(self, output_dir: str = "samples"):
        """
        Create sample input files for testing
        
        Args:
            output_dir: Directory to create sample files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Sample data
        sample_articles = [
            {
                'title': 'Military Tensions Rise in Border Region',
                'subject': 'International Conflict',
                'text': 'Recent reports indicate increased military activity along the disputed border between Country A and Country B. Troops have been mobilized and there have been several skirmishes reported by local sources.'
            },
            {
                'title': 'Economic Trade Agreement Signed',
                'subject': 'Diplomatic Relations',
                'text': 'Representatives from multiple nations gathered today to sign a comprehensive trade agreement that will reduce tariffs and promote economic cooperation.'
            },
            {
                'title': 'Natural Disaster Relief Efforts',
                'subject': 'Humanitarian Aid',
                'text': 'International relief organizations have deployed teams to assist with recovery efforts following the recent earthquake. No military forces are involved in these humanitarian operations.'
            }
        ]
        
        # Create sample TXT file
        txt_content = []
        for article in sample_articles:
            txt_content.append(f"{article['title']}\n{article['text']}")
        
        with open(output_dir / "sample_articles.txt", 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(txt_content))
        
        # Create sample CSV file
        with open(output_dir / "sample_articles.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['title', 'subject', 'text'])
            writer.writeheader()
            writer.writerows(sample_articles)
        
        # Create sample JSON file
        with open(output_dir / "sample_articles.json", 'w', encoding='utf-8') as f:
            json.dump(sample_articles, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Sample files created in: {output_dir}")
        print("   - sample_articles.txt")
        print("   - sample_articles.csv") 
        print("   - sample_articles.json")
