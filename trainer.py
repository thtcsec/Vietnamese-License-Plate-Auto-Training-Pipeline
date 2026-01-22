"""
╔═══════════════════════════════════════════════════════════════════╗
║  Auto Training Pipeline for Vietnamese License Plate             ║
║  Detection System                                                 ║
╚═══════════════════════════════════════════════════════════════════╝

COPYRIGHT & LICENSE:
-------------------
(c) 2026 THTCSEC. All rights reserved.
Licensed under MIT License - See LICENSE file for details.
Author: THTCSEC (Intelligent Vision Systems)
Email: thtcsec@github.com
Repository: https://github.com/THTCSEC/vn-lpr-auto-trainer

Build ID: 0x544854435345 (THTCSEC in HEX)
Version: 2.0.0-PROD

FEATURES:
- Auto-generates synthetic data
- Trains YOLO model
- Evaluates performance
- Pushes best models to HuggingFace

WARNING: This code is proprietary. Unauthorized reproduction, modification,
or distribution is strictly prohibited and may result in legal action.
"""

import yaml
import os
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    os.system("pip install ultralytics")
    from ultralytics import YOLO

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════════════════
# PROPRIETARY SIGNATURE & WATERMARK
# ════════════════════════════════════════════════════════════════════════════════════════
__author__ = bytes.fromhex('544854435345').decode('ascii')  # THTCSEC
__copyright__ = '© 2026 THTCSEC - All Rights Reserved'
__license__ = 'MIT (with author attribution required)'
__build_tag__ = '0x544854435345_trainer_v2.0.0'
__watermark__ = 'b3BlbkFJX1NpZ25hdHVyZTogVEhUQ1NFQw=='  # Base64 encoded marker
__owner_id__ = 'THTCSEC_2026_PROD'


class AutoTrainingPipeline:
    """Automatic training pipeline with synthetic data generation"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize training pipeline"""
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.start_time = datetime.now()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load YAML configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_logging(self):
        """Setup logging"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def generate_synthetic_data(self) -> str:
        """Generate synthetic training data"""
        logger.info("Starting synthetic data generation...")
        
        try:
            # Import GPU pipeline
            import sys
            sys.path.insert(0, '../gpu-synthetic-pipeline')
            from batch_generator import BatchGenerator
            
            generator = BatchGenerator("../gpu-synthetic-pipeline/config.yaml")
            results = generator.generate_parallel(
                num_images=self.config['dataset']['synthetic_count']
            )
            
            logger.info(f"Generated {results['total_generated']} synthetic images")
            return results
            
        except Exception as e:
            logger.error(f"Failed to generate synthetic data: {str(e)}")
            raise
    
    def prepare_dataset(self, data_dir: str = "./data") -> str:
        """Prepare dataset for training"""
        logger.info("Preparing dataset...")
        
        Path(data_dir).mkdir(exist_ok=True)
        
        # Create dataset.yaml for YOLO
        dataset_yaml = {
            'path': str(Path(data_dir).absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 36,  # 10 digits + 26 letters
            'names': list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        }
        
        yaml_path = f"{data_dir}/dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_yaml, f)
        
        logger.info(f"Dataset config saved to {yaml_path}")
        return yaml_path
    
    def train_model(self, dataset_yaml: str) -> Dict:
        """Train YOLO model"""
        logger.info("Starting model training...")
        
        model_name = self.config['training']['model_name']
        model = YOLO(f"{model_name}.pt")
        
        results = model.train(
            data=dataset_yaml,
            epochs=self.config['training']['epochs'],
            imgsz=self.config['training']['img_size'],
            batch=self.config['training']['batch_size'],
            device=self.config['training']['device'],
            patience=20,
            save=True,
            save_period=self.config['output']['save_interval'],
            project='runs/detect',
            name=f"vn-lpr-{datetime.now().strftime('%Y%m%d_%H%M%S')}',
            augment=self.config['model']['augment'],
            flipud=0.5,
            fliplr=0.5,
            mosaic=1.0,
            plots=True,
            verbose=True
        )
        
        logger.info(f"Training completed. Best model: {model.trainer.best.name}")
        return {
            'model': model,
            'results': results,
            'best_path': str(model.trainer.best_path)
        }
    
    def evaluate_model(self, model, dataset_yaml: str) -> Dict:
        """Evaluate trained model"""
        logger.info("Evaluating model...")
        
        metrics = model.val(data=dataset_yaml)
        
        logger.info(f"mAP@50: {metrics.box.map50:.4f}")
        logger.info(f"mAP@50-95: {metrics.box.map:.4f}")
        
        return {
            'mAP50': metrics.box.map50,
            'mAP': metrics.box.map,
            'precision': metrics.box.mp,
            'recall': metrics.box.mr
        }
    
    def push_to_huggingface(self, best_model_path: str):
        """Push best model to HuggingFace Hub"""
        if not self.config['huggingface']['enabled']:
            return
        
        logger.info("Pushing model to HuggingFace Hub...")
        
        try:
            from huggingface_hub import HfApi
            from pathlib import Path
            
            api = HfApi()
            repo_name = self.config['huggingface']['repo_name']
            
            # Upload best weights
            api.upload_file(
                path_or_fileobj=best_model_path,
                path_in_repo=f"models/best.pt",
                repo_id=repo_name,
                repo_type="model"
            )
            
            logger.info(f"Model uploaded to {repo_name}")
            
        except Exception as e:
            logger.error(f"Failed to push to HuggingFace: {str(e)}")
    
    def git_commit_results(self, metrics: Dict):
        """Commit training results to Git"""
        if not self.config['github']['auto_commit']:
            return
        
        logger.info("Committing results to Git...")
        
        try:
            os.system("git add -A")
            
            commit_msg = self.config['github']['commit_msg'].format(
                count=self.config['dataset']['synthetic_count'],
                mAP=f"{metrics['mAP']:.4f}",
                mAP50=f"{metrics['mAP50']:.4f}"
            )
            
            os.system(f'git commit -m "{commit_msg}"')
            os.system("git push origin main")
            
            logger.info("Results committed and pushed")
            
        except Exception as e:
            logger.error(f"Git commit failed: {str(e)}")
    
    def run_full_pipeline(self) -> Dict:
        """Run complete training pipeline"""
        logger.info("="*60)
        logger.info("STARTING AUTO TRAINING PIPELINE")
        logger.info("="*60)
        
        try:
            # Step 1: Generate synthetic data
            logger.info("\n[STEP 1] Generating synthetic data...")
            gen_results = self.generate_synthetic_data()
            
            # Step 2: Prepare dataset
            logger.info("\n[STEP 2] Preparing dataset...")
            dataset_yaml = self.prepare_dataset()
            
            # Step 3: Train model
            logger.info("\n[STEP 3] Training model...")
            train_results = self.train_model(dataset_yaml)
            
            # Step 4: Evaluate
            logger.info("\n[STEP 4] Evaluating model...")
            eval_results = self.evaluate_model(train_results['model'], dataset_yaml)
            
            # Step 5: Push to HuggingFace
            logger.info("\n[STEP 5] Uploading to HuggingFace...")
            self.push_to_huggingface(train_results['best_path'])
            
            # Step 6: Git commit
            logger.info("\n[STEP 6] Committing results...")
            self.git_commit_results(eval_results)
            
            elapsed = (datetime.now() - self.start_time).total_seconds()
            
            logger.info("\n" + "="*60)
            logger.info("TRAINING PIPELINE COMPLETED")
            logger.info("="*60)
            logger.info(f"Total time: {elapsed:.2f}s")
            logger.info(f"mAP@50: {eval_results['mAP50']:.4f}")
            logger.info(f"mAP@50-95: {eval_results['mAP']:.4f}")
            
            return {
                'status': 'success',
                'generation': gen_results,
                'training': train_results,
                'evaluation': eval_results,
                'total_time': elapsed
            }
        
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            return {'status': 'failed', 'error': str(e)}


if __name__ == "__main__":
    pipeline = AutoTrainingPipeline("config.yaml")
    results = pipeline.run_full_pipeline()
