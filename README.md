# MV-SAM3D

MV-SAM3D is a multi-view 3D reconstruction framework that extends SAM 3D Objects to leverage observations from multiple viewpoints. It supports both single-object and multi-object generation, and is designed to produce more stable geometry, texture, and scene-level consistency. 

## Paper

- arXiv: [https://arxiv.org/abs/2603.11633](https://arxiv.org/abs/2603.11633)

## Installation

To use this repository, sam3 and sam3d-objects along with their model checkpoints must be downloaded. depth-anything-v3 is not considered here because my purpose is not related. This will consume a big chunk of your space, so some precautious check should be done on your workspace's disk space to see if you can implement this framework. A docker image will be released soon. 

To clone this repo: 
```bash
git clone git@github.com:NguyenTuanMi/MV-SAM3D.git
cd MV-SAM3D
```

### Prerequisites

- A linux 64-bits architecture (i.e. linux-64 platform in mamba info).
- A NVIDIA GPU with at least 32 Gb of VRAM.

### Installing SAM3D Objects

To install SAMD3D:
```bash
conda env create -f environments/default.yml
conda activate sam3d-objects
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
pip install -e '.[dev]'
pip install -e '.[p3d]'
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install -e '.[inference]'
```

### Getting SAM3D Checkpoints

#### From HuggingFace

First, you need to install the huggingface in the env:
```bash
pip install 'huggingface-hub[cli]<1.0'
```

Secondly, you need to create a huggingface account and request access to the checkpoints on the SAM 3D objects HuggingFace [repo](https://huggingface.co/facebook/sam-3d-objects). Once accepted, you need to be authenticated to download the checkpoints. You can do this by running the following [steps](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication) ((e.g. 'hf auth login' after generating an access token).)

```bash
pip install 'huggingface-hub[cli]<1.0'

TAG=hf
hf download \
  --repo-type model \
  --local-dir checkpoints/${TAG}-download \
  --max-workers 1 \
  facebook/sam-3d-objects
mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG}
rm -rf checkpoints/${TAG}-download
```

### Installing SAM3 

To install SAM3: 
```bash
cd .. 
git clone org-16943930@github.com:facebookresearch/sam3.git
cd sam3
pip install -e .
```

### Getting SAM3 Checkpoints

#### From HuggingFace

Similar to SAM3D-Objects, you need to request access to the checkpoints on the SAM3 HuggingFace [repo](https://huggingface.co/facebook/sam3). After that, you can follow this guide to install:
```bash
cd .. && cd MV-SAM3D
hf download --repo-type model --local-dir checkpoints/${TAG}-download --max-workers 1 facebook/sam3
mv checkpoints/${TAG}-download/sam3.pt checkpoints/${TAG}
rm -rf checkpoints/${TAG}-download
```

## Data Format

```text
scene/
├── images/
│   ├── 0.png
│   ├── 1.png
│   └── ...
├── object_a/
│   ├── 0.png
│   ├── 1.png
│   └── ...
├── object_b/
│   └── ...
└── ...
```

Mask files are RGBA PNG where alpha indicates foreground.

## Default Settings (No Extra Flags)

For single-object inference (`run_inference_weighted.py`), key defaults are:

- Stage 1 weighting: enabled (`stage1_entropy_alpha=30.0`)
- Stage 2 weighting: enabled (`stage2_weight_source=entropy`)
- Stage 2 alpha defaults: `stage2_entropy_alpha=30.0`, `stage2_visibility_alpha=30.0`

## Quickstart

Put images of an object captured in multi view inside the data folder. (e.g: data/your-scene-name/images). 

### Preprocessing with SAM3 to obtain masks 
```bash
python preprocessing/build_mvsam3d_dataset.py --input data/your-scene --objects object-name --sam3_checkpoint checkpoints/hf/sam3.pt 
```

If you use depth anything v3:
```bash
python scripts/run_da3.py \
  --image_dir ./data/your_scene/images \
  --output_dir ./da3_outputs/your_scene
```

### Run inference with SAM3D-Objects 

If you use depth anything v3: 
#### Single-object inference

```bash
python run_inference_weighted.py \
  --input_path ./data/example \
  --mask_prompt stuffed_toy \
  --da3_output ./da3_outputs/example/da3_output.npz
```

#### Multi-object inference

```bash
python run_inference_weighted.py \
  --input_path ./data/desk_objects0 \
  --mask_prompt keyboard,speaker,mug,stuffed_toy \
  --da3_output ./da3_outputs/desk_objects0/da3_output.npz \
  --merge_da3_glb \
  --run_pose_optimization
```

If you don't use depth anything v3 (like I did): 
```bash
python run_inference_weighted.py   --input_path ./data/your-scene  --mask_prompt object-name
```

### Noted
Sometimes, you need to adjust the confidence_threshold inside the 'build_mvsam3d_dataset.py' to obtain masks for all images. 

## Results Comparison

### Single-object

<table>
<tr>
  <td align="center" width="33%"><b>Single-View (View 3)</b></td>
  <td align="center" width="33%"><b>Single-View (View 6)</b></td>
  <td align="center" width="33%"><b>MV-SAM3D</b></td>
</tr>
<tr>
  <td align="center" width="33%" style="padding: 5px;">
    <b>Input Image</b><br>
    <img src="data/example/images/3.png" width="100%" style="max-width: 300px;"/>
  </td>
  <td align="center" width="33%" style="padding: 5px;">
    <b>Input Image</b><br>
    <img src="data/example/images/6.png" width="100%" style="max-width: 300px;"/>
  </td>
  <td align="center" width="33%" style="padding: 5px;">
    <b>Input Images</b><br>
    <table width="100%" cellpadding="2" cellspacing="2">
      <tr>
        <td align="center"><img src="data/example/images/1.png" width="80px"/></td>
        <td align="center"><img src="data/example/images/2.png" width="80px"/></td>
        <td align="center"><img src="data/example/images/3.png" width="80px"/></td>
        <td align="center"><img src="data/example/images/4.png" width="80px"/></td>
      </tr>
      <tr>
        <td align="center"><img src="data/example/images/5.png" width="80px"/></td>
        <td align="center"><img src="data/example/images/6.png" width="80px"/></td>
        <td align="center"><img src="data/example/images/7.png" width="80px"/></td>
        <td align="center"><img src="data/example/images/8.png" width="80px"/></td>
      </tr>
    </table>
  </td>
</tr>
<tr>
  <td align="center" colspan="3">
    <b>↓ 3D Reconstruction ↓</b>
  </td>
</tr>
<tr>
  <td align="center" width="33%" style="padding: 5px;">
    <img src="data/example/visualization_results/view3_cropped.gif" width="100%" style="max-width: 300px;"/>
    <br><sub>Single-view baseline.</sub>
  </td>
  <td align="center" width="33%" style="padding: 5px;">
    <img src="data/example/visualization_results/view6_cropped.gif" width="100%" style="max-width: 300px;"/>
    <br><sub>Single-view baseline.</sub>
  </td>
  <td align="center" width="33%" style="padding: 5px;">
    <img src="data/example/visualization_results/all_views_cropped.gif" width="100%" style="max-width: 300px;"/>
    <br><sub>Better multi-view consistency.</sub>
  </td>
</tr>
</table>

### Multi-object

<table>
<tr>
  <td align="center" width="33%"><b>SAM 3D (single-view)</b></td>
  <td align="center" width="33%"><b>MV-SAM3D w/o Pose Optimization</b></td>
  <td align="center" width="33%"><b>MV-SAM3D (full)</b></td>
</tr>
<tr>
  <td align="center" width="33%" style="padding: 5px;">
    <img src="data/example/visualization_results/laptop_scene_0_sam3d.gif" width="100%" style="max-width: 300px;"/>
    <br><sub>Shape and pose are often unstable.</sub>
  </td>
  <td align="center" width="33%" style="padding: 5px;">
    <img src="data/example/visualization_results/laptop_scene_0_mvsam3d.gif" width="100%" style="max-width: 300px;"/>
    <br><sub>Multi-view improves object quality.</sub>
  </td>
  <td align="center" width="33%" style="padding: 5px;">
    <img src="data/example/visualization_results/laptop_scene_0_mvsam3d_optimized.gif" width="100%" style="max-width: 300px;"/>
    <br><sub>Improved overall scene alignment.</sub>
  </td>
</tr>
</table>

## Quick Start

### Single-object inference

```bash
python run_inference_weighted.py \
  --input_path ./data/example \
  --mask_prompt stuffed_toy \
  --da3_output ./da3_outputs/example/da3_output.npz
```

### Multi-object inference

```bash
python run_inference_weighted.py \
  --input_path ./data/desk_objects0 \
  --mask_prompt keyboard,speaker,mug,stuffed_toy \
  --da3_output ./da3_outputs/desk_objects0/da3_output.npz \
  --merge_da3_glb \
  --run_pose_optimization
```



## Citation

```bibtex
@article{li2026mv,
  title={MV-SAM3D: Adaptive Multi-View Fusion for Layout-Aware 3D Generation},
  author={Li, Baicheng and Wu, Dong and Li, Jun and Zhou, Shunkai and Zeng, Zecui and Li, Lusong and Zha, Hongbin},
  journal={arXiv preprint arXiv:2603.11633},
  year={2026}
}
```

## Acknowledgments

We thank the authors of [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) and [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) for their excellent work!!!

## License

Please refer to [LICENSE](./LICENSE) for usage terms.
