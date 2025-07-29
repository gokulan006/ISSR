"""
Ollama MIE Classifier Integration
"""

class MIEExpertClassifier:
    """Classifies articles using the Ollama MIE Expert LLM."""
    def __init__(self, config):
        self.config = config
    def classify_with_context(self, text, context=None):
        """Classify text with optional RAG context (stub)."""
        return {"classification": "UNKNOWN", "reasoning": "", "action_type": None, "coding": {}} 