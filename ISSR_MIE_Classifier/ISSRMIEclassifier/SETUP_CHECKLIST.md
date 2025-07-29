# ✅ Enhanced MIE Classifier - Setup Checklist

## 🔍 Pre-Installation Verification

### Required Software
- [ ] Python 3.8+ installed (`python --version`)
- [ ] Git installed (`git --version`)
- [ ] pip installed (`pip --version`)
- [ ] Ollama installed (optional) (`ollama --version`)

## 📦 Repository Contents Check

### Core Files
- [ ] `main.py` - Main entry point
- [ ] `requirements.txt` - Python dependencies
- [ ] `setup_ollama.sh` - Ollama setup script
- [ ] `README.md` - Project documentation
- [ ] `QUICKSTART.md` - Setup instructions

### Data Files
- [ ] `data/raw/final_data_true.csv` - Training dataset (2.5MB)
- [ ] `data/raw/MIE_Coding_Instructions.txt` - Coding guidelines
- [ ] `data/raw/2015.csv` through `data/raw/2023.csv` - Yearly data
- [ ] `data/raw/mie_articles_only.csv` - MIE-only articles
- [ ] `data/raw/non_mie_248.csv` - Non-MIE articles

### ML Models
- [ ] `ml/models/enhanced_mie_classifier.py` - Core classifier (501 lines)
- [ ] `ml/models/enhanced_mie_system.py` - Enhanced system
- [ ] `ml/models/hybrid_classifier.py` - Hybrid approach
- [ ] `ml/training/train.py` - Training script
- [ ] `ml/evaluation/evaluate.py` - Evaluation script
- [ ] `ml/evaluation/test_mie_expert.py` - Expert testing

### NLP Components
- [ ] `nlp/preprocessing/text_processor.py` - Text preprocessing
- [ ] `nlp/analysis/sentiment_analyzer.py` - Sentiment analysis
- [ ] `nlp/extraction/entity_extractor.py` - Entity extraction

### RAG Components
- [ ] `rag/context/context_builder.py` - Context building
- [ ] `rag/retrieval/retriever.py` - Document retrieval
- [ ] `rag/vectorstore/embedding_store.py` - Vector embeddings

### Ollama Integration
- [ ] `ollama/integration/mie_classifier.py` - LLM integration
- [ ] `ollama/models/Modelfile` - Model definition
- [ ] `ollama/models/README.md` - Model documentation
- [ ] `ollama/prompts/README.md` - Prompt documentation

### Utility Files
- [ ] `data_manager.py` - Data management utilities
- [ ] `process_all_csvs.py` - CSV processing
- [ ] `setup_cloud_data.py` - Cloud data setup
- [ ] `test_2015_processing.py` - Processing tests

## 🚀 Installation Steps Verification

### 1. Dependencies Installation
```bash
pip install -r requirements.txt
```
- [ ] All packages installed successfully
- [ ] No version conflicts
- [ ] No missing dependencies

### 2. NLTK Data Download
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words')"
```
- [ ] NLTK data downloaded successfully
- [ ] No download errors

### 3. Ollama Setup (Optional)
```bash
# Start Ollama
ollama serve

# Pull model
ollama pull gemma3:latest

# Setup MIE Expert
./setup_ollama.sh
```
- [ ] Ollama service running
- [ ] Gemma3 model downloaded
- [ ] MIE Expert model created
- [ ] Setup script executed successfully

## 🧪 Functionality Tests

### Basic System Test
```bash
python main.py
```
- [ ] System starts without errors
- [ ] Interactive mode loads
- [ ] Can input article text
- [ ] Classification results displayed

### Data Loading Test
```python
from ml.models.enhanced_mie_classifier import EnhancedMIEClassifier
classifier = EnhancedMIEClassifier()
df = classifier.load_and_prepare_data('data/raw/final_data_true.csv')
print(f"Loaded {len(df)} articles")
```
- [ ] Data loads successfully
- [ ] Expected number of articles (496 total)
- [ ] Required columns present

### Ollama Connection Test
```python
if classifier.check_ollama():
    print("✅ Ollama connected")
else:
    print("❌ Ollama not available")
```
- [ ] Ollama connection works (if enabled)
- [ ] Graceful fallback if not available

## 📊 Expected Data Structure

### Training Data (`final_data_true.csv`)
- [ ] 496 total articles
- [ ] 248 MIE articles (label = 1)
- [ ] 248 non-MIE articles (label = 0)
- [ ] Columns: Title, Subject, Text, label

### File Sizes
- [ ] `final_data_true.csv` ≈ 2.5MB
- [ ] Yearly CSV files: 200-360MB each
- [ ] Total data size: ~2GB

## 🎯 Success Criteria

### Minimum Viable Setup
- [ ] Python dependencies installed
- [ ] Training data accessible
- [ ] Basic ML classifier works
- [ ] Interactive mode functional

### Full Feature Setup
- [ ] Ollama integration working
- [ ] RAG system functional
- [ ] Sentiment analysis active
- [ ] Entity extraction working
- [ ] All components integrated

## 🚨 Common Issues & Solutions

### Missing Files
- [ ] Verify all files listed above are present
- [ ] Check file permissions
- [ ] Ensure no corrupted downloads

### Dependency Issues
- [ ] Use virtual environment
- [ ] Update pip: `pip install --upgrade pip`
- [ ] Force reinstall: `pip install -r requirements.txt --force-reinstall`

### Ollama Issues
- [ ] Check if Ollama is running: `curl http://localhost:11434/api/tags`
- [ ] Restart Ollama service
- [ ] Verify model exists: `ollama list`

### Memory Issues
- [ ] Use smaller dataset for testing
- [ ] Increase system memory
- [ ] Process data in batches

---

## ✅ Final Verification

Run this command to test everything:
```bash
python -c "
from ml.models.enhanced_mie_classifier import EnhancedMIEClassifier
import pandas as pd

# Test data loading
classifier = EnhancedMIEClassifier()
df = classifier.load_and_prepare_data('data/raw/final_data_true.csv')
print(f'✅ Data loaded: {len(df)} articles')

# Test Ollama
if classifier.check_ollama():
    print('✅ Ollama connected')
else:
    print('⚠️  Ollama not available (optional)')

# Test basic functionality
X = df['Title'].fillna('') + ' ' + df['Subject '].fillna('') + ' ' + df['Text'].fillna('')
y = df['label']
classifier.train_enhanced_model(X, y)
print('✅ Model training successful')

print('🎉 Setup verification complete!')
"
```

**If all checks pass, your Enhanced MIE Classifier is ready to use! 🚀** 