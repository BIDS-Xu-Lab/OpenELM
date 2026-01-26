
from openelm.tokens_map import TYPE_TOKEN_MAP_DICT
from openelm.utils import load_elm_model, batched_inference_input_generator

import numpy as np
import yaml
import pickle
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for embedding language model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    args = parser.parse_args()

    # read config file
    configs = yaml.safe_load(open(args.config, "r"))

    # check if input_dir exists
    if not os.path.exists(configs['input_dir']):
        raise ValueError(f"Input directory {configs['input_dir']} does not exist")

    # check if output_dir exists, if not, create it
    if not os.path.exists(configs['output_dir']):
        os.makedirs(configs['output_dir'])
        print("Output directory {} created".format(configs['output_dir']))
    else:
        print("Output directory {} already exists".format(configs['output_dir']))

    #########
    ## Load model config, and tokenizer
    #########

    tokenizer, model_config, lora_elm = load_elm_model(configs)    
    emb_token = TYPE_TOKEN_MAP_DICT[model_config.model_type]["emb_tok"]

    #########
    ## Load tasks & perform inference
    #########

    for task in configs['tasks']:
        
        embeddings = {j: np.load(os.path.join(configs['input_dir'], task['embedding_files'][j])) for j in range(len(task['embedding_files']))}
        prompt_template = task['prompt_template']
        
        # Initialize generator for each task
        data_generator = batched_inference_input_generator(prompt_template, emb_token, tokenizer, embeddings, batch_size=configs['batch_size'])
        results = []
        for batch_input_ids, batch_embs, prompt_lengths in data_generator:
            # Generate outputs
            outputs = lora_elm.generate(
                input_ids=batch_input_ids,
                domain_embeddings=batch_embs,
                max_length=configs['max_length'],
                eos_token_id=lora_elm.config.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                repetition_penalty=configs['repetition_penalty']
            )
            # Decode outputs
            for j, output in enumerate(outputs):
                prompt_length = prompt_lengths[j]
                result = tokenizer.decode(output[prompt_length:], skip_special_tokens=True)
                results.append(result)
        
        # Save results
        with open(os.path.join(configs['output_dir'], task['task_name'] + '_inference_results.pkl'), 'wb') as f:
            pickle.dump(results, f)