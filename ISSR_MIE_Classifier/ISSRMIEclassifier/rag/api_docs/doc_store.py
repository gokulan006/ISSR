import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any

try:
	from sentence_transformers import SentenceTransformer
	from sklearn.metrics.pairwise import cosine_similarity
	except_import_error = None
except Exception as e:
	except_import_error = e


class APIDocStore:
	"""Vector store over API documentation for RAG-2 spec retrieval."""
	def __init__(self, model_name: str = 'all-MiniLM-L6-v2', base_path: str = 'data/vectorstores/api_docs'):
		self.model_name = model_name
		self.base_path = Path(base_path)
		self.base_path.mkdir(parents=True, exist_ok=True)
		self.model = SentenceTransformer(model_name) if except_import_error is None else None

	def _get_store_dir(self, api_id: str) -> Path:
		p = self.base_path / api_id
		p.mkdir(parents=True, exist_ok=True)
		return p

	def ingest_openapi(self, api_id: str, openapi: Dict[str, Any]) -> None:
		store_dir = self._get_store_dir(api_id)
		# Flatten paths into text corpus
		texts = []
		records = []
		paths = openapi.get('paths', {})
		for path, ops in paths.items():
			for method, spec in ops.items():
				sumtext = json.dumps({
					'path': path,
					'method': method.upper(),
					'summary': spec.get('summary',''),
					'parameters': [p.get('name') for p in spec.get('parameters', [])],
					'required': [p.get('name') for p in spec.get('parameters', []) if p.get('required')],
					'security': spec.get('security', openapi.get('security', []))
				}, ensure_ascii=False)
				texts.append(sumtext)
				records.append({'path': path, 'method': method.upper(), 'raw': spec})
		embeddings = self.model.encode(texts) if self.model else []
		with open(store_dir / 'docs.pkl', 'wb') as f:
			pickle.dump({'records': records, 'embeddings': embeddings, 'texts': texts}, f)
		with open(store_dir / 'openapi_raw.json', 'w', encoding='utf-8') as f:
			json.dump(openapi, f)

	def get_spec(self, api_id: str, intent: str, top_k: int = 3) -> Dict[str, Any]:
		store_dir = self._get_store_dir(api_id)
		file = store_dir / 'docs.pkl'
		if not file.exists():
			return {}
		with open(file, 'rb') as f:
			data = pickle.load(f)
		records = data['records']
		texts = data['texts']
		embeddings = data['embeddings']
		query_emb = self.model.encode([intent]) if self.model else []
		scores = cosine_similarity(query_emb, embeddings)[0]
		best = scores.argsort()[::-1][:top_k]
		# Prefer GET with clearly named params if possible
		choice = records[best[0]]
		params = []
		for p in choice['raw'].get('parameters', []):
			params.append(p.get('name'))
		security = choice['raw'].get('security', [])
		return {
			'endpoint': choice['path'],
			'method': choice['method'],
			'params': params,
			'auth': security
		}
