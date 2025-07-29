# 🚀 Enhanced MIE Classifier - Quick Setup Guide

## 📋 Prerequisites

Before you begin, ensure you have:

- **Python 3.8+** installed
- **Git** installed
- **pip** package manager
- **Ollama** (for local LLM functionality) - [Download here](https://ollama.ai/)

## 🛠️ Installation Steps

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd GSOC
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Required NLTK Data
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words')"
```

### 4. Setup Ollama (Optional but Recommended)

**Step 4a: Install Ollama**
- Visit [ollama.ai](https://ollama.ai/) and download for your OS
- Follow installation instructions for your platform

**Step 4b: Start Ollama Service**
```bash
# On Windows (PowerShell):
ollama serve

# On macOS/Linux:
ollama serve &
```

**Step 4c: Pull Required Model**
```bash
ollama pull gemma3:latest
```

**Step 4d: Setup MIE Expert Model**
```bash
# On Windows (PowerShell):
./setup_ollama.sh

# On macOS/Linux:
chmod +x setup_ollama.sh
./setup_ollama.sh
```

## 📊 Data Setup

The project comes with pre-processed data:

- **Training Data**: `data/raw/final_data_true.csv` (248 MIE + 248 non-MIE articles)
- **Raw Data**: Multiple CSV files for years 2015-2023
- **MIE Coding Instructions**: `data/raw/MIE_Coding_Instructions.txt`

## 🎯 Running the System

### Option 1: Interactive Mode (Recommended)
```bash
python main.py
```

This starts an interactive session where you can:
- Enter article titles and text
- Get real-time MIE classification
- See confidence scores and reasoning
- View sentiment analysis and entity extraction

### Option 2: Programmatic Usage
```python
from ml.models.enhanced_mie_classifier import EnhancedMIEClassifier

# Initialize classifier
classifier = EnhancedMIEClassifier()

# Load and train on data
df = classifier.load_and_prepare_data('data/raw/final_data_true.csv')
X = df['Title'].fillna('') + ' ' + df['Subject '].fillna('') + ' ' + df['Text'].fillna('')
y = df['label']
classifier.train_enhanced_model(X, y)

# Classify an article
result = classifier.predict_enhanced_mie(
    title="U.S. military conducts airstrikes in Syria",
    subject="Military action",
    text="The United States military launched targeted airstrikes..."
)
print(result)
```

## 🔧 System Components

The Enhanced MIE Classifier combines:

1. **Machine Learning**: TF-IDF + Naive Bayes/Random Forest
2. **RAG (Retrieval Augmented Generation)**: Similar article retrieval
3. **Ollama LLM**: Custom MIE Expert model for classification
4. **Sentiment Analysis**: VADER sentiment scoring
5. **Entity Extraction**: Country names, fatality counts
6. **MIC Heuristics**: Death word analysis and MIE keywords

## 📁 Project Structure

```
GSOC/
├── main.py                          # Main entry point
├── requirements.txt                 # Python dependencies
├── setup_ollama.sh                 # Ollama setup script
├── data/
│   ├── raw/                        # Raw CSV files
│   │   ├── final_data_true.csv     # Training dataset
│   │   ├── 2015.csv - 2023.csv     # Yearly data
│   │   └── MIE_Coding_Instructions.txt
│   ├── processed/                  # Processed outputs
│   └── embeddings/                 # Vector embeddings
├── ml/
│   └── models/
│       └── enhanced_mie_classifier.py  # Core classifier
├── nlp/
│   ├── preprocessing/
│   ├── analysis/
│   └── extraction/
├── rag/
│   ├── context/
│   ├── retrieval/
│   └── vectorstore/
└── ollama/
    ├── integration/
    ├── models/
    └── prompts/
```

## 🚨 Troubleshooting

### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama service
ollama serve
```

### Missing Dependencies
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

### NLTK Data Issues
```bash
python -c "import nltk; nltk.download('all')"
```

### Memory Issues with Large Datasets
- The system works with the provided `final_data_true.csv` (2.5MB)
- For larger datasets, consider processing in batches

## 🎉 Success Indicators

You'll know everything is working when:

1. ✅ `python main.py` starts without errors
2. ✅ Interactive mode accepts article input
3. ✅ Classification results show confidence scores
4. ✅ Ollama integration works (if enabled)

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all prerequisites are installed
3. Ensure data files are in the correct locations
4. Check that Ollama is running (if using LLM features)

---

**Ready to classify MIE articles! 🎯**
