def llama_token_map_generator():
    shared_map = {
        "emb_tok": "<|reserved_special_token_0|>",
        "emb_tok_id": 128002,
        "gen_tok": "<|reserved_special_token_1|>",
        "gen_tok_id": 128003,
        "pad_tok_id": 128009,
    }
    model_names = [
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
    ]
    for model_name in model_names:
        yield model_name, shared_map

def gemma_token_map_generator():
    shared_map = {
        "emb_tok": "<unused0>",
        "emb_tok_id": 6,
        "gen_tok": "<unused1>",
        "gen_tok_id": 7,
        "pad_tok_id": 0,
    }
    model_names = [
        "google/gemma-3-1b-it",
        "google/gemma-3-4b-it",
        "google/gemma-3-12b-it",
        "google/gemma-3-27b-it",
        "google/medgemma-4b-it",
    ]
    for model_name in model_names:
        yield model_name, shared_map

TOKEN_MAP_DICT = dict(llama_token_map_generator())
TOKEN_MAP_DICT.update(dict(gemma_token_map_generator()))

TYPE_TOKEN_MAP_DICT = {
    "llama": next(llama_token_map_generator())[1],
    "gemma3": next(gemma_token_map_generator())[1],
    "gemma3_text": next(gemma_token_map_generator())[1],
}