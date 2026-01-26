import os # control file paths
import argparse # parse command line arguments
import yaml # read yaml file
import pickle # load target texts
import torch # prepare embeddings for model input
import numpy as np # load embeddings
from transformers import AutoTokenizer # load tokenizer for prompt generation
from datasets import Dataset, interleave_datasets # prepare openelm dataset
from openelm.tokens_map import TOKEN_MAP_DICT # load token map for base model

def check_config(data_config):
    """
    This function checks input_dir, output_dir, and base_model in the configuration file.

    data_config: dict
        The configuration file.
    """
    # check if input_dir exists
    if not os.path.exists(data_config["input_dir"]):
        raise ValueError(f"Input directory {data_config['input_dir']} does not exist")

    # check if base_model is supported
    if data_config["base_model"] not in TOKEN_MAP_DICT:
        raise ValueError(f"Base model {data_config['base_model']} not found in TOKEN_MAP_DICT")

    # check if output_dir exists, if not, create it
    if not os.path.exists(data_config["output_dir"]):
        os.makedirs(data_config["output_dir"])
        print("Output directory {} created".format(data_config["output_dir"]))
    else:
        print("Output directory {} already exists".format(data_config["output_dir"]))

def task_specific_generator(prompt_template, tokenizer, emb_token, gen_token, target_texts, embeddings):
    """
    This generator generates required input for the openelm model.

    prompt_template: str
        The prompt template for the task.
    tokenizer: AutoTokenizer
        The tokenizer for the base model.
    emb_token: str
        The token for the embedding placeholder.
    gen_token: str
        The token for the generation placeholder.
    target_texts: list
        The target texts for the task.
    embeddings: list of numpy arrays
        The embeddings for the task.
    """
    for i in range(len(target_texts)):
        chat = [
            {"role": "user", "content": prompt_template.format(emb_token=emb_token)},
            {"role": "assistant", "content": gen_token+target_texts[i]},
        ]
        yield {"input_ids": tokenizer.apply_chat_template(chat), "domain_embeddings": [torch.Tensor(embeddings[num_emb][i]) for num_emb in range(len(embeddings))]}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare data using config file.')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file.')
    args = parser.parse_args()

    # load data config
    with open(args.config, 'r') as file:
        data_config = yaml.safe_load(file)
    
    # check config
    check_config(data_config)

    # load tokenizer & get emb_token and gen_token
    tokenizer = AutoTokenizer.from_pretrained(data_config["base_model"])
    emb_token = TOKEN_MAP_DICT[data_config["base_model"]]["emb_tok"]
    gen_token = TOKEN_MAP_DICT[data_config["base_model"]]["gen_tok"]

    #########
    # Start of data preparation
    #########

    list_of_train_datasets = []
    list_of_validation_datasets = []
    number_of_tasks = len(data_config['tasks'])
    train_ratio = data_config['train_ratio']
    # loop through tasks
    for i in range(number_of_tasks):

        # load target texts
        with open(os.path.join(data_config['input_dir'], data_config['tasks'][i]['target_text_file']), 'rb') as file:
            taget_texts = pickle.load(file)

        # load embeddings, where key is the index of the embedding file
        number_of_embedding_files = len(data_config['tasks'][i]['embedding_files'])
        embeddings = {j: np.load(os.path.join(data_config['input_dir'], data_config['tasks'][i]['embedding_files'][j])) for j in range(number_of_embedding_files)}

        # check if number of rows are matched bewteen embeddings and target texts
        for j in range(number_of_embedding_files):
            if len(taget_texts) != len(embeddings[j]):
                raise ValueError(f"Number of rows in embeddings file {data_config['tasks'][i]['embedding_files'][j]} ({len(embeddings[j])}) does not match number of target texts ({len(taget_texts)})")
        
        number_of_instances = len(taget_texts)
        print("Loaded {} instances for task {}.".format(number_of_instances, data_config['tasks'][i]['task_name']))

        # split train and validation sets
        # if train_ratio is a float, it is the ratio of training examples to total examples
        if train_ratio < 1:
            train_size = int(train_ratio * number_of_instances)
        # if train_ratio is an int, it is the number of training examples
        else:
            train_size = train_ratio
        
        # sample training ids
        training_ids = np.random.choice(number_of_instances, train_size, replace=False)
        # set validation ids to be the remaining ids
        validation_ids = np.setdiff1d(np.arange(number_of_instances), training_ids)

        # extract training and validation target texts and embeddings
        train_taget_texts = [taget_texts[j] for j in training_ids]
        train_embeddings = [embeddings[j][training_ids] for j in range(number_of_embedding_files)]
        validation_taget_texts = [taget_texts[j] for j in validation_ids]
        validation_embeddings = [embeddings[j][validation_ids] for j in range(number_of_embedding_files)]

        # initialize task-specific generator function & convert to dataset
        prompt_template = data_config['tasks'][i]['prompt_template']
        list_of_train_datasets.append(Dataset.from_generator(lambda: task_specific_generator(prompt_template, tokenizer, emb_token, gen_token, train_taget_texts, train_embeddings)))
        list_of_validation_datasets.append(Dataset.from_generator(lambda: task_specific_generator(prompt_template, tokenizer, emb_token, gen_token, validation_taget_texts, validation_embeddings)))
    
    #########
    # End of data preparation
    #########
    
    # if there are multiple tasks (as multiple datasets), interleave the datasets
    if len(list_of_train_datasets) > 1:
        formatted_train_dataset = interleave_datasets(list_of_train_datasets)
        formatted_validation_dataset = interleave_datasets(list_of_validation_datasets)
    # if there is only one task (as one dataset), use the single dataset
    else:
        formatted_train_dataset = list_of_train_datasets[0]
        formatted_validation_dataset = list_of_validation_datasets[0]
    
    formatted_train_dataset.save_to_disk(os.path.join(data_config["output_dir"], "encoded_training_dataset"))
    formatted_validation_dataset.save_to_disk(os.path.join(data_config["output_dir"], "encoded_validation_dataset"))
    print("Encoded training and validation datasets saved to {}".format(data_config["output_dir"]))