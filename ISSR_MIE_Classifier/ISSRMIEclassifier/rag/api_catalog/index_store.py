import os
import pickle
from pathlib import Path
from typing import List, Dict, Any

try:
	from sentence_transformers import SentenceTransformer
	from sklearn.metrics.pairwise import cosine_similarity
	except_import_error = None
except Exception as e:
	except_import_error = e


class APICatalogIndex:
	"""Vector store over API catalog entries for RAG-1 API selection."""
	def __init__(self, model_name: str = 'all-MiniLM-L6-v2', store_path: str = 'data/vectorstores/api_index'):
		self.model_name = model_name
		self.store_path = Path(store_path)
		self.store_path.mkdir(parents=True, exist_ok=True)
		self.model = SentenceTransformer(model_name) if except_import_error is None else None
		self.entries: List[Dict[str, Any]] = []
		self.embeddings = None
		self._load()

	def build_index(self, catalog: List[Dict[str, Any]]) -> None:
		self.entries = catalog
		texts = [f"{e.get('id','')} {e.get('name','')} {e.get('description','')} {' '.join(e.get('tags',[]))}" for e in catalog]
		self.embeddings = self.model.encode(texts) if self.model else []
		self._save()

	def search_best_api(self, text: str, top_k: int = 3) -> List[Dict[str, Any]]:
		if not self.entries or self.embeddings is None:
			return []
		query_emb = self.model.encode([text]) if self.model else []
		scores = cosine_similarity(query_emb, self.embeddings)[0]
		indices = scores.argsort()[::-1][:top_k]
		results = []
		for idx in indices:
			entry = dict(self.entries[idx])
			entry['score'] = float(scores[idx])
			results.append(entry)
		return results

	def _save(self) -> None:
		with open(self.store_path / 'api_index.pkl', 'wb') as f:
			pickle.dump({'entries': self.entries, 'embeddings': self.embeddings}, f)

	def _load(self) -> None:
		file = self.store_path / 'api_index.pkl'
		if file.exists():
			with open(file, 'rb') as f:
				data = pickle.load(f)
				self.entries = data.get('entries', [])
				self.embeddings = data.get('embeddings')
