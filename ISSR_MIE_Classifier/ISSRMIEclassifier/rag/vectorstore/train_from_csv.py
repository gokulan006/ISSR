import argparse
from rag.vectorstore.embedding_store import MIEVectorStore

def main():
	parser = argparse.ArgumentParser(description='Train MIE vectorstore from CSV')
	parser.add_argument('--csv', required=True, help='Path to CSV file')
	parser.add_argument('--out', required=True, help='Output vectorstore directory')
	parser.add_argument('--model', default='all-MiniLM-L6-v2')
	args = parser.parse_args()
	config = {"rag": {"embedding_model": args.model, "vector_store_path": args.out}}
	vs = MIEVectorStore(config)
	vs.train_with_mie_data(args.csv)
	print(f"Vectorstore saved to {args.out}")

if __name__ == '__main__':
	main()
