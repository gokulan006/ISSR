import os
import json
import requests
from urllib.parse import urljoin
from typing import Dict, Any

DEFAULT_TIMEOUT = 12

class SafeAPIClient:
	"""Thin wrapper around requests with simple domain whitelist and auth helpers."""
	def __init__(self, allowed_domains=None):
		self.allowed_domains = set(allowed_domains or [])

	def _is_allowed(self, url: str) -> bool:
		return any(d in url for d in self.allowed_domains)

	def call(self, base_url: str, endpoint: str, method: str, params: Dict[str, Any], auth: Any = None, headers: Dict[str, str] = None):
		url = urljoin(base_url if base_url.endswith('/') else base_url + '/', endpoint.lstrip('/'))
		if self.allowed_domains and not self._is_allowed(url):
			return {'status': 403, 'text': 'Domain not allowed'}
		s = requests.Session()
		hdrs = dict(headers or {})
		# Simple auth support
		if isinstance(auth, list) and auth:
			# OpenAPI security array; rely on env vars e.g. AUTH_BEARER
			bearer = os.getenv('AUTH_BEARER')
			if bearer:
				hdrs['Authorization'] = f'Bearer {bearer}'
		elif isinstance(auth, dict) and auth.get('type') == 'Bearer':
			bearer = os.getenv(auth.get('env', 'AUTH_BEARER'), '')
			if bearer:
				hdrs['Authorization'] = f'Bearer {bearer}'
		try:
			if method.upper() == 'GET':
				r = s.get(url, params=params, headers=hdrs, timeout=DEFAULT_TIMEOUT)
			else:
				r = s.post(url, json=params, headers=hdrs, timeout=DEFAULT_TIMEOUT)
			ct = (r.headers.get('Content-Type') or '').lower()
			if 'application/json' in ct:
				return {'status': r.status_code, 'json': r.json()}
			return {'status': r.status_code, 'text': r.text}
		except Exception as e:
			return {'status': 500, 'text': str(e)}
