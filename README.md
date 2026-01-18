# Occlusion Robustness Evaluation for SAM/MedSAM

Evaluate the robustness of SAM and MedSAM to occlusion in medical image segmentation.

## Project Structure
PROJECT2_OCCLUSIONSAM/
├── configs/           # Configuration files
├── data/              # Dataset loaders
├── models/            # Model wrappers
├── occlusion/         # Occlusion techniques
├── evaluation/        # Metrics
├── utils/             # Utilities
├── outputs/           # Results
├── dataset/           # Your datasets
├── checkpoints/       # Model weights
└── main.py            # Entry point

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Project2_OcclusionSAM
```

2. Create and activate virtual environment:

**Using venv:**
```bash
python -m venv occlusion_sam
source occlusion_sam/bin/activate  # On Windows: occlusion_sam\Scripts\activate
```

**Using conda:**
```bash
conda create -n occlusion_sam python=3.10
conda activate occlusion_sam
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download model checkpoints:

**SAM ViT-B:**
```bash
# Using wget
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O checkpoints/sam_vit_b.pth

# Using curl
curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -o checkpoints/sam_vit_b.pth
```

**MedSAM:**
```bash
# Using wget
wget https://zenodo.org/records/10689643/files/medsam_vit_b.pth -O checkpoints/medsam_vit_b.pth

# Using curl
curl -L https://zenodo.org/records/10689643/files/medsam_vit_b.pth -o checkpoints/medsam_vit_b.pth
```

5. Prepare datasets in dataset/ folder:

**CVC-ClinicDB:**
- Download from: https://www.kaggle.com/datasets/balraj98/cvcclinicdb
- Extract to `dataset/CVC-ClinicDB/`
- Structure:
  ```
  dataset/CVC-ClinicDB/
  ├── Original/       # Original images
  └── Ground Truth/   # Segmentation masks
  ```

**Kvasir-SEG:**
- Download from: https://datasets.simula.no/kvasir-seg/
- Extract to `dataset/Kvasir-SEG/`
- Structure:
  ```
  dataset/Kvasir-SEG/
  ├── images/        # Original images
  └── masks/         # Segmentation masks
  ```

## Usage

Basic usage:
# No occlusion (baseline)
python main.py --dataset cvc --model medsam --occlusion none

# Cutout with single ratio
python main.py --dataset kvasir --model sam --occlusion cutout --ratio 0.2

# Cutmix with multiple ratios
python main.py --dataset cvc --model medsam --occlusion cutmix --ratio 0.1 0.2 0.3 0.4 0.5

With visualization:
python main.py --dataset cvc --model medsam --occlusion cutout --ratio 0.2 --visualize

Test on subset:
python main.py --dataset kvasir --model sam --occlusion none --num_samples 10
