import requests

def generate(prompt: str, model: str = 'mie-expert', url: str = 'http://localhost:11434') -> str:
	try:
		resp = requests.post(f"{url}/api/generate", json={"model": model, "prompt": prompt, "stream": False}, timeout=30)
		if resp.status_code == 200:
			return resp.json().get('response', '').strip()
		return ''
	except Exception:
		return ''
