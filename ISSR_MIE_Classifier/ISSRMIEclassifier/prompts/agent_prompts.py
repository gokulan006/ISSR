def build_master_prompt(article: str, question: str, api_choice: dict, api_response: dict) -> str:
	live_ctx = []
	if 'json' in api_response:
		live_ctx.append(str(api_response['json'])[:1500])
	elif 'text' in api_response:
		live_ctx.append(api_response['text'][:1500])
	live = "\n".join(live_ctx)
	api_name = api_choice.get('id') if api_choice else 'N/A'
	return f"""
You are an expert analyst. Answer the user's question by combining the news article with the live data.

ARTICLE:\n{article[:4000]}

LIVE DATA (from {api_name}):\n{live}

QUESTION:\n{question}

Answer concisely. Cite facts from the live data where relevant.
"""
