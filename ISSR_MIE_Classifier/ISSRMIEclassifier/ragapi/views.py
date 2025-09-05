from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

# Modular RAG using rag/vectorstore/embedding_store.py
try:
	from rag.vectorstore.embedding_store import MIEVectorStore
except Exception:
	MIEVectorStore = None

# In-model RAG using ml/models/enhanced_mie_classifier.py
try:
	from ml.models.enhanced_mie_classifier import EnhancedMIEClassifier
except Exception:
	EnhancedMIEClassifier = None

# Agent orchestration
try:
	from agent.orchestrator import run_pipeline
	from rag.api_catalog.index_store import APICatalogIndex
	from rag.api_docs.doc_store import APIDocStore
except Exception:
	run_pipeline = None
	APICatalogIndex = None
	APIDocStore = None

# Keep singletons for simplicity
_VECTOR_STORE = None
_CLASSIFIER = None
_API_INDEX = APICatalogIndex() if APICatalogIndex else None
_API_DOCS = APIDocStore() if APIDocStore else None

@csrf_exempt
def rag_modular_train(request):
	if request.method != 'POST':
		return JsonResponse({'error': 'POST required'}, status=405)
	if MIEVectorStore is None:
		return JsonResponse({'error': 'MIEVectorStore unavailable'}, status=500)
	try:
		payload = json.loads(request.body.decode('utf-8') or '{}')
		config = payload.get('config', {
			'rag': {
				'embedding_model': 'all-MiniLM-L6-v2',
				'vector_store_path': 'data/vectorstores/mie'
			}
		})
		data_path = payload.get('data_path', 'data/raw/final_data_true.csv')
		global _VECTOR_STORE
		_VECTOR_STORE = MIEVectorStore(config)
		_VECTOR_STORE.train_with_mie_data(data_path)
		return JsonResponse({'status': 'ok'})
	except Exception as e:
		return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def rag_modular_query(request):
	if request.method != 'POST':
		return JsonResponse({'error': 'POST required'}, status=405)
	if _VECTOR_STORE is None:
		return JsonResponse({'error': 'Vector store not initialized. Call /api/rag/modular/train first.'}, status=400)
	try:
		payload = json.loads(request.body.decode('utf-8') or '{}')
		query = payload.get('query', '')
		k = int(payload.get('top_k', 5))
		results = _VECTOR_STORE.search(query, top_k=k)
		return JsonResponse({'results': results})
	except Exception as e:
		return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def rag_model_init(request):
	if request.method != 'POST':
		return JsonResponse({'error': 'POST required'}, status=405)
	if EnhancedMIEClassifier is None:
		return JsonResponse({'error': 'EnhancedMIEClassifier unavailable'}, status=500)
	try:
		payload = json.loads(request.body.decode('utf-8') or '{}')
		model_name = payload.get('model_name', 'mie-expert')
		data_path = payload.get('data_path', 'data/raw/final_data_true.csv')
		global _CLASSIFIER
		_CLASSIFIER = EnhancedMIEClassifier(model_name=model_name)
		df = _CLASSIFIER.load_and_prepare_data(data_path)
		_CLASSIFIER.train_enhanced_model(
			(df['Title'].fillna('') + ' ' + df['Subject '].fillna('') + ' ' + df['Text'].fillna('')).tolist(),
			df['label']
		)
		_CLASSIFIER.create_rag_embeddings(df)
		return JsonResponse({'status': 'ok', 'num_articles': int(len(df))})
	except Exception as e:
		return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def rag_model_similar(request):
	if request.method != 'POST':
		return JsonResponse({'error': 'POST required'}, status=405)
	if _CLASSIFIER is None:
		return JsonResponse({'error': 'Classifier not initialized. Call /api/rag/model/init first.'}, status=400)
	try:
		payload = json.loads(request.body.decode('utf-8') or '{}')
		text = payload.get('text', '')
		k = int(payload.get('top_k', 3))
		results = _CLASSIFIER.retrieve_similar_articles(text, top_k=k)
		return JsonResponse({'results': results})
	except Exception as e:
		return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def rag_model_classify(request):
	if request.method != 'POST':
		return JsonResponse({'error': 'POST required'}, status=405)
	if _CLASSIFIER is None:
		return JsonResponse({'error': 'Classifier not initialized. Call /api/rag/model/init first.'}, status=400)
	try:
		payload = json.loads(request.body.decode('utf-8') or '{}')
		title = payload.get('title', '')
		subject = payload.get('subject', '')
		text = payload.get('text', '')
		result = _CLASSIFIER.predict_enhanced_mie(title, subject, text)
		return JsonResponse({'result': result})
	except Exception as e:
		return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def agent_run(request):
	if request.method != 'POST':
		return JsonResponse({'error': 'POST required'}, status=405)
	if run_pipeline is None:
		# attempt to surface import error details
		try:
			import importlib
			importlib.import_module('agent.orchestrator')
		except Exception as e:
			return JsonResponse({'error': 'Agent orchestrator unavailable', 'detail': str(e)}, status=500)
	try:
		payload = json.loads(request.body.decode('utf-8') or '{}')
		article = payload.get('article', '')
		question = payload.get('question', '')
		base_url_map = payload.get('base_url_map', {})
		whitelist = payload.get('whitelist', [])
		dry_run = bool(payload.get('dry_run', False))
		result = run_pipeline(article, question, base_url_map=base_url_map, whitelist=whitelist, dry_run=dry_run)
		return JsonResponse(result)
	except Exception as e:
		return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def rag_model_classify_json(request):
	if request.method != 'POST':
		return JsonResponse({'error': 'POST required'}, status=405)
	if _CLASSIFIER is None:
		# lazy init
		try:
			payload = json.loads(request.body.decode('utf-8') or '{}')
			data_path = payload.get('data_path', 'data/raw/final_data_true.csv')
			global _CLASSIFIER
			_CLASSIFIER = EnhancedMIEClassifier()
			df = _CLASSIFIER.load_and_prepare_data(data_path)
			df['combined_text'] = df['Title'].fillna('') + ' ' + df['Subject '].fillna('') + ' ' + df['Text'].fillna('')
			X = df['combined_text']
			y = df['label']
			_CLASSIFIER.train_enhanced_model(X, y)
			_CLASSIFIER.create_rag_embeddings(df)
		except Exception as e:
			return JsonResponse({'error': f'classifier init failed: {e}'}, status=500)
	try:
		payload = json.loads(request.body.decode('utf-8') or '{}')
		title = payload.get('title', '')
		subject = payload.get('subject', '')
		text = payload.get('text', '')
		result = _CLASSIFIER.predict_enhanced_mie(title, subject, text)
		# Prefer LLM JSON if present
		oa = result.get('ollama_analysis')
		if isinstance(oa, dict) and 'coding' in oa:
			out = {
				'classification': oa.get('classification','UNKNOWN'),
				'reasoning': oa.get('reasoning',''),
				'codeable': oa.get('codeable', True),
				'coding': oa.get('coding', {}),
				'countries_involved': oa.get('countries', []),
				'missing_fields': oa.get('missing_fields', []),
				'notes': oa.get('notes',''),
				'confrontation_key': oa.get('confrontation_key','')
			}
			return JsonResponse(out)
		# Fallback JSON
		entities = result.get('entities', {'countries': []})
		sentiment = result.get('sentiment_analysis') or result.get('sentiment') or {}
		fallback = _CLASSIFIER.build_coding_json_fallback(title, subject, text, entities, sentiment, result.get('final_prediction',0))
		return JsonResponse(fallback)
	except Exception as e:
		return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def agent_catalog_ingest(request):
	if request.method != 'POST':
		return JsonResponse({'error': 'POST required'}, status=405)
	if _API_DOCS is None:
		return JsonResponse({'error': 'Doc store unavailable'}, status=500)
	try:
		payload = json.loads(request.body.decode('utf-8') or '{}')
		api_id = payload.get('api_id')
		openapi = payload.get('openapi')
		if not api_id or not openapi:
			return JsonResponse({'error': 'api_id and openapi required'}, status=400)
		_API_DOCS.ingest_openapi(api_id, openapi)
		return JsonResponse({'status': 'ok'})
	except Exception as e:
		return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def agent_catalog_list(request):
	if request.method != 'GET':
		return JsonResponse({'error': 'GET required'}, status=405)
	# For now, read from existing index entries; build from YAML if empty
	try:
		apis = getattr(_API_INDEX, 'entries', []) if _API_INDEX else []
		if not apis and _API_INDEX:
			import yaml
			from pathlib import Path
			cfg = Path('config/api_catalog.yml')
			if cfg.exists():
				data = yaml.safe_load(cfg.read_text(encoding='utf-8')) or {}
				catalog = data.get('apis', [])
				_API_INDEX.build_index(catalog)
				apis = catalog
		return JsonResponse({'apis': apis})
	except Exception as e:
		return JsonResponse({'error': str(e)}, status=500)
