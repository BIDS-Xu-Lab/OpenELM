import argparse
from transformers import AutoTokenizer
from openelm.model import initialize_embedding_model_from_causal_lm
from time import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Initialize embedding language model from causal LM.')
    parser.add_argument('--base_model', type=str, required=True, help='Base model to initialize embedding model from.')
    parser.add_argument('--dim_embed_domain', type=int, required=True, help='Dimension of the embedding domain.')
    parser.add_argument('--dim_adapter_hidden', type=int, required=True, help='Dimension of the adapter hidden layer.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory to save the embedding model.')
    args = parser.parse_args()

    # initialize embedding model
    start_time = time()
    model = initialize_embedding_model_from_causal_lm(
        args.base_model,
        args.dim_embed_domain,
        args.dim_adapter_hidden
    )
    end_time = time()
    print(f"Time taken to initialize embedding model: {end_time - start_time} seconds")

    # get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # save model and tokenizer
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Embedding model and tokenizer saved to {args.output_dir}")