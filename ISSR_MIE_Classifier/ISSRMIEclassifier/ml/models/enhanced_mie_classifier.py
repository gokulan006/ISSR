import pandas as pd
import numpy as np
import requests
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from sentence_transformers import SentenceTransformer
import pickle
import warnings
warnings.filterwarnings('ignore')

class EnhancedMIEClassifier:
    def __init__(self, ollama_url="http://localhost:11434", model_name="MICclass"):
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.ml_pipeline = None
        self.rag_embeddings = None
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Enhanced MIE keywords based on your coding instructions
        self.mie_keywords = {
            'threats': ['threat', 'warn', 'declare war', 'blockade', 'occupy'],
            'military_actions': ['attack', 'clash', 'bombing', 'shelling', 'invasion'],
            'force_display': ['show of force', 'mobilization', 'alert', 'fortify'],
            'border_actions': ['border violation', 'cross border', 'territory'],
            'seizure': ['seize', 'capture', 'detain', 'confiscate'],
            'occupation': ['occupy', 'control territory', 'hold territory'],
            'official_forces': ['military', 'army', 'navy', 'air force', 'government forces']
        }
        
        # Death-related keywords from your existing work
        self.death_keywords = {
            "killed", "dead", "fatalities", "deaths", "massacre", "bombing",
            "casualties", "wounded", "injured", "victims", "perished"
        }
        
    def check_ollama(self):
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            return response.status_code == 200
        except:
            return False
    
    def query_ollama(self, prompt):
        """Query Ollama with a prompt"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                return None
                
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def extract_entities(self, text):
        """Enhanced entity extraction based on your existing work"""
        try:
            # Tokenize and POS tag
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            
            # Named Entity Recognition
            named_entities = ne_chunk(pos_tags)
            
            # Extract countries and fatalities
            countries = []
            fatalities = []
            
            # Look for country names and fatality numbers
            for chunk in named_entities:
                if hasattr(chunk, 'label'):
                    if chunk.label() == 'GPE':  # Geographical Political Entity
                        countries.append(' '.join([c[0] for c in chunk]))
            
            # Extract fatality numbers using regex
            fatality_patterns = [
                r'(\d+)\s*(?:people|soldiers|civilians|troops)\s*(?:killed|dead|died)',
                r'(?:killed|dead|died)\s*(\d+)\s*(?:people|soldiers|civilians|troops)',
                r'(\d+)\s*(?:casualties|fatalities|deaths)',
                r'(?:casualties|fatalities|deaths)\s*(\d+)'
            ]
            
            for pattern in fatality_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                fatalities.extend([int(match) for match in matches])
            
            return {
                'countries': list(set(countries)),
                'fatalities': fatalities,
                'total_fatalities': sum(fatalities) if fatalities else 0
            }
        except Exception as e:
            print(f"Entity extraction error: {e}")
            return {'countries': [], 'fatalities': [], 'total_fatalities': 0}
    
    def analyze_sentiment_and_death_words(self, text):
        """Enhanced sentiment and death word analysis from your work"""
        # VADER sentiment analysis
        sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
        
        # Count death-related words
        text_lower = text.lower()
        death_word_count = sum(1 for word in self.death_keywords if word in text_lower)
        
        # Count MIE-related keywords
        mie_word_count = 0
        for category, keywords in self.mie_keywords.items():
            mie_word_count += sum(1 for keyword in keywords if keyword in text_lower)
        
        return {
            'sentiment_compound': sentiment_scores['compound'],
            'sentiment_negative': sentiment_scores['neg'],
            'death_word_count': death_word_count,
            'mie_word_count': mie_word_count,
            'is_negative': sentiment_scores['compound'] < -0.3,
            'has_death_words': death_word_count >= 1,
            'has_mie_words': mie_word_count >= 2
        }
    
    def create_rag_embeddings(self, articles_df):
        """Create RAG embeddings for similarity search"""
        print("Creating RAG embeddings...")
        
        # Combine text features
        articles_df['combined_text'] = (
            articles_df['Title'].fillna('') + ' ' + 
            articles_df['Subject '].fillna('') + ' ' + 
            articles_df['Text'].fillna('')
        )
        
        # Create embeddings
        texts = articles_df['combined_text'].tolist()
        embeddings = self.sentence_model.encode(texts)
        
        self.rag_embeddings = {
            'embeddings': embeddings,
            'articles': articles_df,
            'texts': texts
        }
        
        print(f"Created embeddings for {len(texts)} articles")
    
    def retrieve_similar_articles(self, query_text, top_k=3):
        """Retrieve similar articles using RAG"""
        if self.rag_embeddings is None:
            return []
        
        # Encode query
        query_embedding = self.sentence_model.encode([query_text])
        
        # Calculate similarities
        similarities = np.dot(self.rag_embeddings['embeddings'], query_embedding.T).flatten()
        
        # Get top-k similar articles
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        similar_articles = []
        for idx in top_indices:
            article = self.rag_embeddings['articles'].iloc[idx]
            similar_articles.append({
                'title': article['Title'],
                'subject': article['Subject '],
                'text': article['Text'][:200] + "...",
                'label': article['label'],
                'similarity': similarities[idx]
            })
        
        return similar_articles
    
    def create_mie_prompt_with_rag(self, title, subject, text, similar_articles):
        """Create enhanced prompt with RAG examples"""
        
        # Build examples from similar articles
        examples = ""
        for i, article in enumerate(similar_articles, 1):
            label = "MIE" if article['label'] == 1 else "NOT_MIE"
            examples += f"""
Example {i}:
Title: {article['title']}
Subject: {article['subject']}
Text: {article['text']}
Classification: {label}
---
"""
        
        prompt = f"""
You are an expert at identifying Militarized Interstate Events (MIEs) in news articles.

MIE Definition: A militarized incident is an explicit, non-routine, and governmentally authorized action by official military forces or government representatives of one state against another state.

Here are similar examples from the training data:

{examples}

Now analyze this new article:

TITLE: {title}
SUBJECT: {subject}
TEXT: {text[:800]}...

Based on the examples above and MIE criteria, classify this article as:
- "MIE" if it describes a militarized interstate event
- "NOT_MIE" if it does not

Also provide:
1. Reasoning for your classification
2. What type of MIE action (if any): threat, show of force, attack, etc.
3. Countries involved (if identifiable)
4. Any fatalities mentioned

Response format:
Classification: [MIE/NOT_MIE]
Reasoning: [explanation]
Action Type: [type of military action if MIE]
Countries: [list of countries]
Fatalities: [number if mentioned]
"""
        return prompt
    
    def load_and_prepare_data(self, filepath=None):
        """Load and prepare the dataset"""
        print("Loading dataset...")
        
        # Use data manager if available
        try:
            from data_manager import DataManager
            dm = DataManager()
            self.df = dm.load_training_data()
        except ImportError:
            # Fallback to direct file loading
            if filepath is None:
                filepath = 'data/raw/final_data_true.csv'
            self.df = pd.read_csv(filepath)
        
        # Convert Probable MIE to binary labels
        self.df['label'] = (self.df['Probable MIE'] == 1).astype(int)
        
        print(f"Dataset: {self.df.shape}")
        print(f"MIE articles (1): {(self.df['label'] == 1).sum()}")
        print(f"Non-MIE articles (0): {(self.df['label'] == 0).sum()}")
        
        # Create RAG embeddings
        self.create_rag_embeddings(self.df)
        
        return self.df
    
    def train_enhanced_model(self, X, y):
        """Train enhanced ML model"""
        print("Training enhanced ML model...")
        
        # TF-IDF vectorization
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        # Train multiple models
        X_tfidf = self.vectorizer.fit_transform(X)
        
        # Naive Bayes (from your existing work)
        self.nb_model = MultinomialNB()
        self.nb_model.fit(X_tfidf, y)
        
        # Random Forest (enhanced)
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_model.fit(X_tfidf, y)
        
        print("Enhanced models trained!")
    
    def predict_enhanced_mie(self, title, subject, text):
        """Enhanced MIE prediction combining all approaches"""
        
        # 1. Entity extraction
        entities = self.extract_entities(f"{title} {subject} {text}")
        
        # 2. Sentiment and keyword analysis
        sentiment_analysis = self.analyze_sentiment_and_death_words(f"{title} {subject} {text}")
        
        # 3. ML prediction
        combined_text = f"{title} {subject} {text}"
        X_test = self.vectorizer.transform([combined_text])
        
        nb_pred = self.nb_model.predict(X_test)[0]
        rf_pred = self.rf_model.predict(X_test)[0]
        
        # 4. RAG retrieval
        similar_articles = self.retrieve_similar_articles(combined_text, top_k=3)
        
        # 5. Ollama prediction with RAG
        ollama_prompt = self.create_mie_prompt_with_rag(title, subject, text, similar_articles)
        ollama_response = self.query_ollama(ollama_prompt)
        
        # 6. Parse Ollama response
        ollama_analysis = self.parse_ollama_response(ollama_response)
        
        # 7. Ensemble decision
        ml_votes = [nb_pred, rf_pred]
        ml_decision = 1 if sum(ml_votes) >= 1 else 0
        
        final_decision = self.combine_predictions(
            ml_decision, ollama_analysis, sentiment_analysis, entities
        )
        
        return {
            'final_prediction': final_decision,
            'ml_prediction': ml_decision,
            'ollama_analysis': ollama_analysis,
            'entities': entities,
            'sentiment_analysis': sentiment_analysis,
            'similar_articles': similar_articles,
            'confidence': self.calculate_confidence(ml_decision, ollama_analysis, sentiment_analysis)
        }
    
    def parse_ollama_response(self, response):
        """Parse structured response from Ollama"""
        if not response:
            return {'classification': 'UNKNOWN', 'reasoning': 'No response'}
        
        # Extract information from response
        classification = 'UNKNOWN'
        reasoning = ''
        action_type = ''
        countries = []
        fatalities = 0
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('Classification:'):
                classification = line.split(':')[1].strip()
            elif line.startswith('Reasoning:'):
                reasoning = line.split(':')[1].strip()
            elif line.startswith('Action Type:'):
                action_type = line.split(':')[1].strip()
            elif line.startswith('Countries:'):
                countries_str = line.split(':')[1].strip()
                countries = [c.strip() for c in countries_str.split(',') if c.strip()]
            elif line.startswith('Fatalities:'):
                try:
                    fatalities = int(line.split(':')[1].strip())
                except:
                    fatalities = 0
        
        return {
            'classification': classification,
            'reasoning': reasoning,
            'action_type': action_type,
            'countries': countries,
            'fatalities': fatalities
        }
    
    def combine_predictions(self, ml_decision, ollama_analysis, sentiment_analysis, entities):
        """Combine predictions from all sources"""
        
        # Weight the different sources
        ml_weight = 0.4
        ollama_weight = 0.4
        heuristic_weight = 0.2
        
        # ML score
        ml_score = ml_decision
        
        # Ollama score
        ollama_score = 1 if ollama_analysis['classification'] == 'MIE' else 0
        
        # Heuristic score based on your existing work
        heuristic_score = 0
        if sentiment_analysis['is_negative'] and sentiment_analysis['has_mie_words']:
            heuristic_score = 1
        elif entities['total_fatalities'] > 0 and sentiment_analysis['has_death_words']:
            heuristic_score = 0.8
        
        # Weighted combination
        final_score = (
            ml_score * ml_weight +
            ollama_score * ollama_weight +
            heuristic_score * heuristic_weight
        )
        
        return 1 if final_score >= 0.5 else 0
    
    def calculate_confidence(self, ml_decision, ollama_analysis, sentiment_analysis):
        """Calculate confidence in prediction"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence if multiple sources agree
        if ollama_analysis['classification'] != 'UNKNOWN':
            confidence += 0.2
        
        if sentiment_analysis['has_mie_words']:
            confidence += 0.1
        
        if sentiment_analysis['is_negative']:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def interactive_mode(self):
        """Interactive classification mode"""
        print("\n=== ENHANCED MIE CLASSIFICATION SYSTEM ===")
        print("Combines: ML + RAG + Ollama + Your MIC Heuristics")
        
        while True:
            print("\n" + "-"*50)
            title = input("Article Title (or 'quit'): ").strip()
            if title.lower() == 'quit':
                break
            
            subject = input("Subject: ").strip()
            text = input("Text: ").strip()
            
            if not any([title, subject, text]):
                print("Please provide some content!")
                continue
            
            print("\n🔍 Analyzing with Enhanced System...")
            result = self.predict_enhanced_mie(title, subject, text)
            
            # Display results
            print(f"\n📊 ENHANCED ANALYSIS RESULTS:")
            print(f"Final Prediction: {'MIE' if result['final_prediction'] == 1 else 'NOT_MIE'}")
            print(f"Confidence: {result['confidence']:.2f}")
            
            print(f"\n🤖 ML Prediction: {'MIE' if result['ml_prediction'] == 1 else 'NOT_MIE'}")
            
            if result['ollama_analysis']['classification'] != 'UNKNOWN':
                print(f"🧠 Ollama Analysis:")
                print(f"  Classification: {result['ollama_analysis']['classification']}")
                print(f"  Reasoning: {result['ollama_analysis']['reasoning']}")
                print(f"  Action Type: {result['ollama_analysis']['action_type']}")
                print(f"  Countries: {result['ollama_analysis']['countries']}")
                print(f"  Fatalities: {result['ollama_analysis']['fatalities']}")
            
            print(f"\n🔍 Entity Extraction:")
            print(f"  Countries: {result['entities']['countries']}")
            print(f"  Total Fatalities: {result['entities']['total_fatalities']}")
            
            print(f"\n📈 Sentiment Analysis:")
            print(f"  Sentiment: {result['sentiment_analysis']['sentiment_compound']:.3f}")
            print(f"  MIE Words: {result['sentiment_analysis']['mie_word_count']}")
            print(f"  Death Words: {result['sentiment_analysis']['death_word_count']}")
            
            if result['similar_articles']:
                print(f"\n📚 Similar Articles (RAG):")
                for i, article in enumerate(result['similar_articles'][:2], 1):
                    print(f"  {i}. {article['title'][:50]}... (Similarity: {article['similarity']:.3f})")

def main():
    print("🚀 ENHANCED MIE CLASSIFICATION SYSTEM")
    print("Building on your MIC work + RAG + LLM")
    
    # Initialize enhanced classifier
    classifier = EnhancedMIEClassifier()
    
    # Check Ollama
    if classifier.check_ollama():
        print("✅ Ollama connected!")
    else:
        print("❌ Ollama not available. Using ML + RAG only.")
    
    # Load and prepare data
    df = classifier.load_and_prepare_data('final_data_true.csv')
    
    # Prepare features
    df['combined_text'] = df['Title'].fillna('') + ' ' + df['Subject '].fillna('') + ' ' + df['Text'].fillna('')
    X = df['combined_text']
    y = df['label']
    
    # Train enhanced model
    classifier.train_enhanced_model(X, y)
    
    # Start interactive mode
    classifier.interactive_mode()

if __name__ == "__main__":
    main() 