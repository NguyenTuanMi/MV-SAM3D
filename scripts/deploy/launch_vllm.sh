#!/bin/bash
set -e
if [[ -z "$CONDA_DEFAULT_ENV" || "$CONDA_DEFAULT_ENV" != "qwen_mass_estimator" ]]; then
    echo "No conda environment is activated or conda environment is not corrected."
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate qwen_mass_estimator
else
    echo "Conda environment '$CONDA_DEFAULT_ENV' is activated."
fi  

VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000