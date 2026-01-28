<h1 align="center">
OpenELM
</h1>

<p align="center">
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</p>

# 🖥️ Environment Setup
```bash
# create a fresh environment
conda create -n your_env_name python=3.11 
conda activate your_env_name
# install required packages
pip install -r requirements.txt
```

# 🚀 Quick Start

1. Download demo_dataset.zip from the [latest release](https://github.com/BIDS-Xu-Lab/OpenELM/releases/download/v1.0/demo_dataset.zip) and unzip it into the project root directory.
2. Adapt Llama model into ELM using demonstration dataset in `demo_dataset` folder (take less than 1 hour if using single H100):
    ```bash
    bash run_demo.sh
    ```
    💡 This bash script is a running example and wraps the following three steps: Data Preparation → Model Initialization → Training. Please go documentation for detailed usage.
3. Perform inference using trained ELM
    ```bash
    python inference.py --config configs/inference_config_demo.yaml
    ```

# 📖 Documentation

## Step 1: Prepare Dataset

Prepare formatted datasets from your raw data (embeddings and target texts) using configuration files.

```bash
python prepare_dataset.py --config configs/your_data_config.yaml
```

**Config structure:**
- `base_model`: Base model identifier (e.g., `meta-llama/Llama-3.1-8B-Instruct`, `google/gemma-3-1b-it`)
- `input_dir`: Directory containing your raw data files
- `output_dir`: Directory where formatted dataset will be saved
- `train_ratio`: Ratio of training examples (float) or number of training examples (int)
- `tasks`: List of tasks, each with:
  - `task_name`: Name of the task
  - `target_text_file`: File containing target texts (pickle format)
  - `embedding_files`: List of embedding files (numpy format)
  - `prompt_template`: Template with `{emb_token}` placeholders

**Example config:**`configs/prepare_dataset_config_demo.yaml`

**Supported base models:**
- `google/gemma-3-1b-it`
- `google/gemma-3-4b-it`
- `google/gemma-3-12b-it`
- `google/gemma-3-27b-it`
- `google/medgemma-4b-it`
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`

## Step 2: Initialize Model

Initialize embedding language models from base causal language models.

```bash
# Initialize a specific model (requires sufficient memory)
python initialize_model.py \
    --base_model chosen_base_model \
    --dim_embed_domain 1024 \
    --dim_adapter_hidden 2048 \
    --output_dir /path/to/output/dir
```

**Parameters:**
- `--base_model`: HuggingFace model identifier
- `--dim_embed_domain`: Dimension of the embedding domain (e.g., 1024)
- `--dim_adapter_hidden`: Dimension of the adapter hidden layer (e.g., 2048)
- `--output_dir`: Directory to save the initialized model

💡 Model initialization requires significant memory. Consider use the accelerare to estinate required memory:
```bash
accelerate estimate-memory base_model
```

## Step 3: Train Model

Train the ELM model using the prepared dataset:

```bash
accelerate launch --mixed_precision=bf16 --num_processes=1 train.py \
    --datahome /path/to/prepared/dataset \
    --basemodel_path /path/to/initialized/model \
    --output_dir /path/to/output \
    --batch_size 4 --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 --num_train_epochs 5 \
    --eval_steps 100 --save_steps 0.25
```

Training scripts support both single (set `num_processes=1`) and multi-GPU (set `num_processes=2` if 2 GPUs) setups.

**Training parameters:**
- `--datahome`: Path to the prepared dataset directory from Step 1
- `--basemodel_path`: Path to the initialized model from Step 2
- `--output_dir`: Directory to save training checkpoints
- `--batch_size`: Batch size per GPU (default: 4)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 8)
- `--learning_rate`: Learning rate (default: 5e-5)
- `--num_train_epochs`: Number of training epochs (default: 5)
- `--eval_steps`: Steps between evaluations (default: 100)
- `--save_steps`: Steps between saving checkpoints (default: 0.25)

## Step 4: Inference

Perform inference using trained models.

```bash
# Run inference with a specific config
python inference.py --config configs/your_inference_config.yaml
```

**Config structure:**
- `backbone_model_path`: Path to the initialized model from Step 2
- `peft_model_id`: Path to the trained LoRA checkpoint
- `device`: Device to use (e.g., `"cuda"`)
- `input_dir`: Directory containing input embeddings
- `output_dir`: Directory to save inference results
- `tasks`: List of inference tasks with `task_name`, `embedding_files`, and `prompt_template`
- `batch_size`: Batch size for inference (default: 16)
- `repetition_penalty`: Repetition penalty (default: 1.2)
- `max_length`: Maximum generation length (default: 512)

**Example config:** `configs/inference_config_demo.yaml`

**Output:** Results are saved as pickle files in the output directory: `{task_name}_inference_results.pkl`

# ⚙️ Adapting to Your Use Case

1. **Custom Dataset:** Create a config file following `configs/data_config_example.yaml` format
2. **Different Base Model:** Ensure the model is supported in `openelm/tokens_map.py`
3. **Training Configuration:** Use `train.py` directly with your parameters
4. **Inference Tasks:** Create custom inference configs for your specific tasks

# 📝 Citation
If you find OpenELM helpful, please star our repo and cite us:
```bibtex
@misc{ondov2026ctelmdecodingmanipulatingembeddings,
      title={ctELM: Decoding and Manipulating Embeddings of Clinical Trials with Embedding Language Models}, 
      author={Brian Ondov and Chia-Hsuan Chang and Yujia Zhou and Mauro Giuffrè and Hua Xu},
      year={2026},
      eprint={2601.18796},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.18796}, 
}
```

We gratefully acknowledge the Google Research [ELM paper](https://openreview.net/forum?id=qoYogklIPz) for inspiration and foundational ideas.