import json
from typing import Dict, Any

from rag.api_catalog.index_store import APICatalogIndex
from rag.api_docs.doc_store import APIDocStore
from nlp.extraction.api_param_extractor import extract_params
from agent.api_client import SafeAPIClient
from prompts.agent_prompts import build_master_prompt
from ollama.integration.agent_llm import generate

# Simple in-memory singletons
_api_index = APICatalogIndex()
_api_docs = APIDocStore()
_http = SafeAPIClient(allowed_domains=None)  # set via run args


def run_pipeline(article: str, question: str, base_url_map: Dict[str, str] = None, whitelist=None, dry_run: bool = False) -> Dict[str, Any]:
	global _http
	if whitelist:
		_http = SafeAPIClient(allowed_domains=whitelist)
	# RAG-1 choose API
	candidates = _api_index.search_best_api(f"{article}\n{question}", top_k=3)
	api_choice = candidates[0] if candidates else None
	if not api_choice:
		prompt = build_master_prompt(article, question, {'id': 'none'}, {'text': 'No live data available'})
		answer = generate(prompt)
		return {"api_choice": None, "api_spec": {}, "api_response": {}, "final_answer": answer, "trace": {"steps": ["LLM"], "missing_fields": []}}
	# RAG-2 get spec
	spec = _api_docs.get_spec(api_choice.get('id'), question)
	if not spec:
		prompt = build_master_prompt(article, question, api_choice, {'text': 'No live data available'})
		answer = generate(prompt)
		return {"api_choice": api_choice, "api_spec": {}, "api_response": {}, "final_answer": answer, "trace": {"steps": ["RAG1","LLM"], "missing_fields": []}}
	# Params
	params = extract_params(article, question, spec)
	if params['missing']:
		return {"error": "missing_parameters", "missing": params['missing'], "api_spec": spec}
	# HTTP
	api_id = api_choice.get('id')
	base_url = (base_url_map or {}).get(api_id, '')
	if dry_run:
		api_resp = {"status": 200, "json": {"dry_run": True, "spec": spec, "params": params['values']}}
	else:
		api_resp = _http.call(base_url, spec['endpoint'], spec['method'], params['values'], auth=spec.get('auth'))
	# LLM
	prompt = build_master_prompt(article, question, api_choice, api_resp)
	answer = generate(prompt)
	return {
		"api_choice": api_choice,
		"api_spec": spec,
		"api_request": {"base_url": base_url, "endpoint": spec['endpoint'], "method": spec['method'], "params": params['values']},
		"api_response": {"status": api_resp.get('status'), "raw": api_resp.get('json', api_resp.get('text'))},
		"final_answer": answer,
		"trace": {"steps": ["RAG1","RAG2","PARAMS","HTTP","LLM"], "missing_fields": []}
	}
