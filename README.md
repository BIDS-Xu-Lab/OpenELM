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
    💡 This bash script is a running example and wraps the following three steps: Data Preparation → Model Initialization → Training. Please go our [Wiki page](https://github.com/BIDS-Xu-Lab/OpenELM/wiki) for detailed usage/documentation.
3. Perform inference using trained ELM
    ```bash
    python inference.py --config configs/inference_config_demo.yaml
    ```

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