#!/bin/bash

# Data Preparation: prepare dataset
python prepare_dataset.py --config configs/prepare_dataset_config_demo.yaml

# Model Initialization: initialize base elm model
python initialize_model.py --base_model meta-llama/Llama-3.1-8B-Instruct \
--dim_embed_domain 1024 \
--dim_adapter_hidden 2048 \
--output_dir demo_dataset/base_elm_Llama-3.1-8B-Instruct

# Training: train base elm model
accelerate launch --mixed_precision=bf16 --num_processes=1 train.py --datahome demo_dataset/llama_2tasks_dataset \
--basemodel_path demo_dataset/base_elm_Llama-3.1-8B-Instruct \
--output_dir demo_dataset/base_elm_Llama-3.1-8B-Instruct_lora \
--batch_size 4 --gradient_accumulation_steps 8 --learning_rate 5e-5 --num_train_epochs 1 \
--eval_steps 100 --save_steps 0.5