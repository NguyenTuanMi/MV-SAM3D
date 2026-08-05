#!/bin/bash
set -e
if [[ -z "$CONDA_DEFAULT_ENV" || "$CONDA_DEFAULT_ENV" != "sam3d-objects" ]]; then
    echo "No conda environment is activated or conda environment is not corrected."
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate sam3d-objects
else
    echo "Conda environment '$CONDA_DEFAULT_ENV' is activated."
fi  

echo "Conda environment '$CONDA_DEFAULT_ENV' is activated successfully."

# Preposessing with SAM3 and DepthAnything3
CUDA_VISIBLE_DEVICES=1 python preprocessing/build_mvsam3d_dataset.py --input data/$1 --objects test_$1 --sam3_checkpoint checkpoints/hf/sam3.pt 
CUDA_VISIBLE_DEVICES=1 python scripts/run_da3.py --image_dir data/$1/images/ --output_dir da3_outputs/$1 --model_path depth-anything/DA3NESTED-GIANT-LARGE

# Running the inference
INFER_LOG=$(mktemp)
CUDA_VISIBLE_DEVICES=1 python run_inference_weighted.py --input_path data/$1 --mask_prompt test_$1 --da3_output da3_outputs/$1/da3_output.npz | tee "$INFER_LOG"

GLB_PATH=$(grep -oP '(?<=✓ GLB file saved to: ).*' "$INFER_LOG" | tail -1)
if [[ -z "$GLB_PATH" ]]; then
  echo "Could not find GLB output path in inference log — aborting post-processing."
  exit 1
fi

# Use the first available image of the object as the mass-estimation reference
IMAGE_PATH=$(find "data/$1/images" -type f \( -iname "*.png" -o -iname "*.jpg" \) | head -1)

echo "Post-processing GLB: $GLB_PATH"
echo "Reference image:     $IMAGE_PATH"

# Running the post processing layers
CUDA_VISIBLE_DEVICES=1 python run_post_process.py --glb_path "$GLB_PATH" --image_path "$IMAGE_PATH" --name test_$1 --category "$1" --width_m $2 --height_m $3 --depth_m $4 --estimate_mass --export_sim_ready

rm -f "$INFER_LOG"