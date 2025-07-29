"""
Context Builder for RAG System
"""

class ContextBuilder:
    """Builds context for LLM from retrieved documents."""
    def __init__(self, config):
        self.config = config
    def build(self, retrieved_docs):
        """Build context string (stub)."""
        return "" 