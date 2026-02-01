# Quantized Hidden State Prediction in Neural Sequence Models

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green)

## Abstract
This repository contains the code and data for my master's thesis on predicting quantized hidden states in sequence models. We propose a quantized version of the [PHi layer](https://arxiv.org/abs/2503.13431) which uses a continuous information bottleneck. The original PHi layer tends to overestimate the information content in the latent representation because of the use of arbitrary number of Gaussian distributions. We demonstrate that this approach addresses the information overestimation and achieves tighter bounds on information content of the latent representations.

## Authors

* **Suuraj Perpeli** — Università della svizzera italiana and Friedrich Alexander Universität
* **Supervisor:** Prof. Jürgen Schmidhuber — Università della svizzera italiana
* **Co-supervisor:** Vincent Herrmann — Università della svizzera italiana

---

## Repository Overview

The codebase is organized as follows:

```text
├── assets/                                      # Figures and diagrams for the README 
├── data/                                        # data required for experiments
├── quantized_hidden_state_prediction/           # Main package source code
│   ├── configs/                                 # YAML config files for experiments
│   ├── dataset_classes/                         # Dataset wrappers and loaders
│   ├── evaluation/                              # Evaluation scripts and metric implementations
│   ├── models/                                  # Model architectures
│   ├── modules/                                 # Reusable model components (layers, blocks)
│   └── utils/                                   # Helper utilities (IO, logging, metrics)
├── requirements.txt                             # Python dependencies
├── README.md                                    # Project documentation
└── LICENSE                                      # Project license
```

## Key Results
1. We find tighter information bounds when comparing raw PHi loss values.
| Bottleneck | Mem. Seq. | Mem. Prog. | ICLL | Random |
|---|---:|---:|---:|---:|
| Continuous | 8.875 ± 0.4785 | 8.759 ± 0.5512 | 17.003 ± 1.1685 | 8.036 ± 2.5420 |
| VQ (Ours) | 0.712 ± 0.0696 | 1.168 ± 0.0691 | 2.0197 ± 0.3919 | 0.231 ± 0.1372 |
| RQ (Ours) | 0.825 ± 0.0310 | 1.965 ± 0.0386 | 3.310 ± 0.0334 | 1.250 ± 0.0867 |

In addition, the performance in terms of both NLL and PHi losses.
<div style="display:flex; gap:1rem; align-items:flex-start;">
    <figure style="flex:1; margin:0;">
        <img src="assets/pfa_levels_nll-gumbel.pdf" alt="PFA levels NLL (Gumbel)" style="width:100%; height:auto;" />
        <figcaption style="text-align:center; font-size:0.9em; margin-top:0.5rem;">(a) pfa_levels_nll-gumbel.pdf</figcaption>
    </figure>
    <figure style="flex:1; margin:0;">
        <img src="assets/pfa_levels_phi.pdf" alt="PFA levels PHi" style="width:100%; height:auto;" />
        <figcaption style="text-align:center; font-size:0.9em; margin-top:0.5rem;">(b) pfa_levels_phi.pdf</figcaption>
    </figure>
</div>


## Requirements
Python: 3.10+ (Tested on 3.10)
PyTorch: 2.5+
Huggingface account logged in with API keys to download the datasets, tokenizers and pretrained models. For the experiments, approved licence for gated repositories [Llama 3.2](https://huggingface.co/collections/meta-llama/metas-llama-32-language-models-and-evals). 


## Quick Start
1. Clone the repository
```Bash 
git clone https://github.com/suurajroshan/quantized-hidden-state-prediction.git
cd quantized-hidden-state-prediction
```
2. Set up the environment
It is recommended to use a virtual environment:
```Bash
# Create and activate venv
python -m venv venv
source venv/bin/activate  

# Install dependencies
pip install -r requirements.txt
```
3. Run Training
We provide configuration files to replicate the main experiments in the thesis.

### Specialized Models (Trained from scratch)
Train a 12-layer Transformer (100M parameters) on the in-context language learning task.
```Bash
cd quantized_hidden_state_prediction
python training_script.py config_file=configs/llama_0.1B_PHi_residual-gumbel-quantizer.yaml
```

### Pretrained Llama 3.2 (3B parameters)
```Bash
python training_script.py config_file=configs/llama_3B_PHi_residual-gumbel-quantizer.yaml
```

### SLURM (HPC) Usage:
If running on a cluster, use the provided sbatch script as a template:
```Bash
sbatch run_slurm.sh
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
I want to thank Vincent Herrmann for making his code available.