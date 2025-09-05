import re
from typing import Dict, Any, List

# Very simple heuristics; can be replaced by LLM extraction

def extract_params(article: str, question: str, spec: Dict[str, Any]) -> Dict[str, Any]:
	params = {}
	missing: List[str] = []
	for p in spec.get('params', []):
		val = None
		if p in ('ticker', 'ticker_symbol', 'symbol'):
			m = re.search(r'\b([A-Z]{1,5})\b', article)
			if m:
				val = m.group(1)
		elif p in ('lat','lon','latitude','longitude'):
			m = re.search(r'(-?\d{1,2}\.\d{1,8})[,\s]+(-?\d{1,3}\.\d{1,8})', article)
			if m:
				if p in ('lat','latitude'):
					val = m.group(1)
				else:
					val = m.group(2)
		# Add more heuristics as needed
		if val is None:
			missing.append(p)
		else:
			params[p] = val
	return {"values": params, "missing": missing}
