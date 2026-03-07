# 🤖 Vietnamese License Plate Auto-Training Pipeline

**By thtcsec** | MIT License

> **Automated daily training pipeline that generates synthetic data, trains YOLO models, and publishes to HuggingFace Hub**

![Status](https://img.shields.io/badge/status-active-brightgreen)
![Author](https://img.shields.io/badge/author-thtcsec-blue)
![Python](https://img.shields.io/badge/python-3.8+-blue)
![YOLO](https://img.shields.io/badge/YOLO-v8-red)
![License](https://img.shields.io/badge/license-MIT-green)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blue)

## 🎯 Problem Solved

**The Challenge:**
- Manual model retraining is tedious & time-consuming
- Models become stale without fresh training data
- Hard to track training experiments and results
- Difficult to publish and share trained models

**Our Solution:**
- ⏰ **Fully automated** - Runs daily at scheduled time
- 📊 **End-to-end pipeline** - Data generation → Training → Evaluation → Publishing
- 🤗 **HuggingFace integration** - Auto-publish best models
- 📈 **Tracked metrics** - Log all results for comparison
- 🔄 **Git integration** - Auto-commit results
- 📱 **Notifications** - Slack alerts on completion
- 🌐 **GitHub Actions** - No server needed, runs on GitHub infrastructure

## ⚡ What It Does

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Step 1: Generate Synthetic Data                       │
│  └─ 50,000 Vietnamese license plate images             │
│     (using GPU acceleration)                           │
│                                                         │
│  Step 2: Train YOLO Model                              │
│  └─ 100 epochs on synthetic + validation data          │
│                                                         │
│  Step 3: Evaluate Performance                          │
│  └─ mAP@50, mAP@50-95, Precision, Recall               │
│                                                         │
│  Step 4: Upload Best Model                             │
│  └─ Publish to HuggingFace Hub for download            │
│                                                         │
│  Step 5: Commit Results                                │
│  └─ Auto-commit metrics to GitHub                      │
│                                                         │
│  Step 6: Send Notification                             │
│  └─ Slack alert with training results                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Manual Training (Local)
```bash
git clone https://github.com/thtcsec/vn-lpr-auto-trainer
cd vn-lpr-auto-trainer

# Install dependencies
pip install -r requirements.txt

# Ensure GPU synthetic pipeline is available
# (copy or pip install vn-lpr-gpu-synthetic)

# Run full pipeline
python trainer.py

# Results in:
# - runs/detect/vn-lpr-*/weights/best.pt  (best model)
# - runs/detect/vn-lpr-*/results.csv       (metrics)
# - logs/training_*.log                    (detailed logs)
```

### Auto Training (GitHub Actions)

1. **Setup HuggingFace Token**
   - Go to https://huggingface.co/settings/tokens
   - Create new token (write permission)

2. **Add GitHub Secret**
   - Repo → Settings → Secrets and Variables → Actions
   - Add secret: `HF_TOKEN` = your token

3. **Enable Workflows**
   - Go to Actions tab
   - Enable "Auto Daily LPR Training" workflow

4. **Wait for Daily Run**
   - Runs at 2 AM UTC daily
   - Or manually trigger via Actions tab
   - Check results in artifacts

## 📊 Features

### ✅ Fully Automated Pipeline

| Stage | Action | Time |
|-------|--------|------|
| **1. Data Gen** | GPU synthesis of 50K images | ~5 min |
| **2. Dataset Prep** | Split train/val/test | <1 min |
| **3. Training** | YOLO8n for 100 epochs | ~2 hours |
| **4. Evaluation** | Compute all metrics | ~10 min |
| **5. Upload** | Push to HuggingFace Hub | ~5 min |
| **6. Git Commit** | Push results to repo | <1 min |
| **Total** | Full pipeline | **~2.5 hours** |

### 📈 Tracked Metrics

```yaml
mAP@50:      # Mean Average Precision at IoU=0.50
mAP@50-95:   # Mean Average Precision (COCO metric)
Precision:   # How many detections are correct
Recall:      # How many objects are found
Loss/Train:  # Training loss curve
Loss/Val:    # Validation loss curve
```

### 🤗 HuggingFace Integration

**Auto-published to:**
```
https://huggingface.co/thtcsec/vn-lpr-models

Files:
- best.pt          (best model weights)
- README.md        (model card)
- results.csv      (training metrics)
- model_info.json  (metadata)
```

### 📱 Notifications

**Slack Integration (Optional)**
```
✅ VN-LPR Auto Training Completed
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Results:
  mAP@50:    0.925
  mAP@50-95: 0.782
  Precision: 0.94
  Recall:    0.91

⏱️  Time: 2h 35m
🚀 Model uploaded to HuggingFace
📦 Results committed to GitHub
```

## 🔧 Configuration

### Edit `config.yaml`

```yaml
training:
  model_name: "yolov8n"        # nano, small, medium, large
  epochs: 100                  # Training epochs
  batch_size: 32               # Batch size
  img_size: 640                # Image resolution
  device: 0                    # GPU device ID

dataset:
  auto_generate_synthetic: true
  synthetic_count: 50000       # Images to generate
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1

huggingface:
  enabled: true
  repo_name: "vn-lpr-models"
  push_best_model: true

github:
  auto_commit: true
  branch: "main"
  commit_msg: "Auto: Daily training with {count} synthetic images"

schedule:
  enabled: true
  frequency: "daily"
  time: "02:00"  # UTC
```

### Disable Auto Features

```yaml
huggingface:
  enabled: false    # Don't upload to HF

github:
  auto_commit: false  # Don't auto-commit

schedule:
  enabled: false    # Don't schedule
```

## 📂 Output Structure

```
runs/detect/
├── vn-lpr-20260122_020000/
│   ├── weights/
│   │   ├── best.pt           # Best model
│   │   ├── last.pt           # Last checkpoint
│   │   └── epoch*.pt         # Intermediate
│   ├── results.csv           # Metrics
│   ├── results.png           # Charts
│   ├── confusion_matrix.png  # Confusion matrix
│   └── val_*.jpg             # Validation images

logs/
├── training_20260122_020000.log  # Detailed log
└── ...
```

## 🎯 Model Performance

### Expected Results
```
Model: YOLO8n
Dataset: 50K Vietnamese LPR synthetic images
Training: 100 epochs, batch 32

Results:
┌─────────────────┬────────────┐
│ Metric          │ Value      │
├─────────────────┼────────────┤
│ mAP@50          │ 0.920      │
│ mAP@50-95       │ 0.785      │
│ Precision       │ 0.940      │
│ Recall          │ 0.910      │
│ Training Time   │ 2h 30m     │
└─────────────────┴────────────┘
```

### Try Different Models
```yaml
training:
  model_name: "yolov8n"  # Nano    - fastest, less accurate
  model_name: "yolov8s"  # Small   - balanced
  model_name: "yolov8m"  # Medium  - slower, more accurate
  model_name: "yolov8l"  # Large   - very slow
  model_name: "yolov8x"  # XLarge  - slowest, best accuracy
```

## 🌐 GitHub Actions Workflow

### File: `.github/workflows/daily-training.yml`

**Triggers:**
- ⏰ Daily at 2 AM UTC
- 🔄 Manual via `workflow_dispatch`
- 🔗 Can trigger from other workflows

**Environment:**
- **OS**: Ubuntu Latest
- **Runner**: GitHub-hosted (CPU-only by default)
- **Storage**: 100 GB
- **Time Limit**: 6 hours

**Note**: GitHub Actions runners have limited GPU, so:
- Generation uses CPU (slower but works)
- Training uses CPU (expect ~4-5 hours for 100 epochs)
- Consider self-hosted GPU runner for production

### Enable GPU (Optional - Self-hosted)

```bash
# On your local machine with GPU:
mkdir actions-runner
cd actions-runner

# Download and setup
./config.sh --url https://github.com/thtcsec/vn-lpr-auto-trainer \
            --token YOUR_RUNNER_TOKEN

# Run
./run.sh
```

## 📊 Monitoring Training

### Local Training
```bash
# During training, view logs
tail -f logs/training_*.log

# After training, view metrics
tensorboard --logdir=runs/detect
# Open: http://localhost:6006
```

### GitHub Actions Training
```bash
1. Go to Actions tab
2. Click on workflow run
3. Expand job steps
4. Check logs in real-time

# Download results
1. Scroll to "Artifacts"
2. Download "training-results"
3. Extract and analyze
```

## 🔗 Integration with GPU Synthetic Pipeline

### Link the Two Repos

```bash
# The trainer imports the GPU synthetic pipeline via sys.path:
# trainer.py already does:
#   sys.path.insert(0, '../gpu-synthetic-pipeline')
#   from batch_generator import BatchGenerator

# So just ensure both repos are cloned side-by-side:
git clone https://github.com/thtcsec/vn-lpr-auto-trainer    auto-training-pipeline/
git clone https://github.com/thtcsec/vn-lpr-gpu-synthetic   gpu-synthetic-pipeline/
```

## 🐛 Troubleshooting

### GitHub Actions Fails with CUDA Error
```
Error: CUDA not available on runner

Solution:
- GitHub runners use CPU
- Use self-hosted runner with GPU
- Or accept slower training on CPU
```

### HuggingFace Upload Fails
```
Check:
1. HF_TOKEN is set correctly
2. Token has write permission
3. Internet connection active
4. HF account has model creation quota
```

### Model Performance Too Low
```
Check:
1. Synthetic data quality
2. Training hyperparameters
3. Dataset size (try 100K+ images)
4. Model size (try yolov8m or larger)
5. Training epochs (try 200+)
```

## 📈 Roadmap

- [x] Auto data generation
- [x] YOLO training
- [x] HuggingFace integration
- [x] GitHub Actions workflow
- [ ] Multi-GPU distributed training
- [ ] Real-time inference API
- [ ] Model versioning/rollback
- [ ] Performance tracking dashboard
- [ ] Custom alerting (Discord, Teams)
- [ ] Integration with wandb.ai

## 🤝 Integration Examples

### Use Trained Model
```python
from ultralytics import YOLO

# Load from local
model = YOLO('runs/detect/vn-lpr-*/weights/best.pt')

# Load from HuggingFace
model = YOLO('thtcsec/vn-lpr-models')

# Inference
results = model.predict(source='image.jpg')
```

### Continuous Integration with Your Project
```yaml
# Your .github/workflows/your-workflow.yml
- name: Train VN-LPR Model
  run: |
    git clone https://github.com/thtcsec/vn-lpr-auto-trainer
    cd vn-lpr-auto-trainer
    python trainer.py
```

## 📜 License

MIT License - see LICENSE file

## 🔗 Related Projects

- 🎨 [Vietnamese License Plate GPU Synthesis Engine](https://github.com/thtcsec/vn-lpr-gpu-synthesis)
- 🎯 [YOLO Detection](https://github.com/ultralytics/ultralytics)
- 🤗 [HuggingFace Hub](https://huggingface.co/)

## 💬 Support & Contact

- **GitHub Issues**: [Report bugs](https://github.com/thtcsec/vn-lpr-auto-trainer/issues)
- **GitHub Discussions**: [Ask questions](https://github.com/thtcsec/vn-lpr-auto-trainer/discussions)
- **Author**: thtcsec

---

**⭐ Star this project if it helped you!**

*Made with ❤️ for the Vietnamese AI community*
