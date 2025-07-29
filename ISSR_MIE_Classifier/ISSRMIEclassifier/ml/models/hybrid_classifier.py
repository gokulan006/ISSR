"""
Hybrid ML Classifier for MIE System
"""

class HybridClassifier:
    """Combines LLM and traditional ML for final classification."""
    def __init__(self, config):
        self.config = config
    def classify(self, text, ollama_result, context=None):
        """Combine LLM and ML results (stub)."""
        return ollama_result 