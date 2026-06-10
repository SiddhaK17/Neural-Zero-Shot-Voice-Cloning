import logging
import os
import sys

# Use this line ONLY if you are inside train.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# 🛑 WINDOWS CRASH FIX (The "Dummy" Handler)
# ==============================================================================
# The Coqui Trainer tries to create 'trainer_0_log.txt', which causes WinError 32.
# We define a fake handler that accepts the arguments but does absolutely nothing.
class DummyFileHandler(logging.NullHandler):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def emit(self, record):
        pass
    def write(self, *args, **kwargs):
        pass

# Apply the patch immediately to stop the file creation
logging.FileHandler = DummyFileHandler
# ==============================================================================

from TTS.tts.utils.text.tokenizer import TTSTokenizer
import shutil
import time
import datetime
import psutil
import torch
import multiprocessing
import numpy as np

# Import Coqui TTS Libraries
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.utils.audio import AudioProcessor
from TTS.tts.models.glow_tts import GlowTTS

# 🔥 NEW IMPORT: Necessary to load the actual data before training
from TTS.tts.datasets import load_tts_samples

# Import the Tensorboard Logger to save data safely
from trainer.logging.tensorboard_logger import TensorboardLogger

# ==============================================================================
# 🛑 SECTION 1: HARDWARE & SAFETY DIAGNOSTICS
# ==============================================================================

def check_system_health():
    print("\n" + "█"*60)
    print("      🛡️  INITIALIZING SAFETY PROTOCOLS & HARDWARE CHECK      ")
    print("█"*60)
    
    if not torch.cuda.is_available():
        print("❌ CRITICAL FAILURE: No NVIDIA GPU Detected.")
        sys.exit(1)
    
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ GPU ONLINE: {gpu_name}")

    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"ℹ️  VRAM DETECTED: {total_vram:.2f} GB")

    if total_vram >= 7.5:
        print("🚀 PERFORMANCE MODE: High VRAM detected (RTX 4070/3080/4090).")
        batch_size = 16
    elif total_vram >= 5.5:
        print("🛡️  STANDARD MODE: Medium VRAM detected (RTX 3060/4060).")
        batch_size = 12
    else:
        print("⚠️  SAFE MODE: Low VRAM detected (<6GB).")
        batch_size = 4

    free_space = shutil.disk_usage(os.getcwd()).free / 1e9
    print(f"💾 DISK SPACE: {free_space:.2f} GB Free")
    
    battery = psutil.sensors_battery()
    if battery and not battery.power_plugged:
        print("⚠️  POWER WARNING: Laptop is running on BATTERY.")
        time.sleep(3)

    print("✅ SYSTEM CHECK COMPLETE. STARTING ENGINE...")
    print("="*60 + "\n")
    return batch_size

# ==============================================================================
# 🚀 SECTION 3: THE TRAINING ENGINE
# ==============================================================================

def main():
    SAFE_BATCH_SIZE = check_system_health()

    # ==============================================================================
    # ⚙️ SECTION 2: PATHS & CONFIGURATION
    # ==============================================================================

    PROJECT_ROOT = os.getcwd()
    DATA_PATH = os.path.join(PROJECT_ROOT, "LJSpeech-1.1")

    # Generate unique ID
    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    OUTPUT_PATH = os.path.join(PROJECT_ROOT, "proof_logs", f"run_{run_id}")

    # 1. Dataset Configuration
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech", 
        meta_file_train="metadata.csv", 
        path=DATA_PATH
    )

    # 2. Model Configuration
    config = GlowTTSConfig(
        batch_size=SAFE_BATCH_SIZE,
        eval_batch_size=8,
        num_loader_workers=0,        # CPU cores for loading data
        num_eval_loader_workers=0,
        mixed_precision=True, 
        epochs=100, 
        text_cleaner="phoneme_cleaners",
        use_phonemes=True, 
        phoneme_language="en-us",
        phoneme_cache_path=os.path.join(OUTPUT_PATH, "phoneme_cache"),
        run_eval=True, 
        test_delay_epochs=-1,
        print_step=25, 
        print_eval=True, 
        output_path=OUTPUT_PATH, 
        save_step=1000, 
        save_checkpoints=True, 
        save_n_checkpoints=3, 
        save_best_after=1000, 
        test_sentences=[
            "This is a verified proof of the model training progress.",
            "I am training this neural network on my local NVIDIA GPU.",
            "Deep learning allows for high quality voice cloning and synthesis.",
            "The quick brown fox jumps over the lazy dog.",
            "Generative AI is the future of computer science."
        ]
    )

    if not os.path.exists(os.path.join(DATA_PATH, "metadata.csv")):
        print(f"\n❌ ERROR: LJ Speech Dataset not found at {DATA_PATH}")
        return

    # 3. 🔥 NEW: LOAD THE DATA SAMPLES
    # The previous error happened because we didn't run this step.
    # This reads the metadata.csv and creates the list of files to train on.
    print("⏳ LOADING DATASET FILES... (This might take a moment)")
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )
    print(f"✅ DATA LOADED: {len(train_samples)} Training Samples, {len(eval_samples)} Eval Samples.")

    # 4. Initialize Audio & Model
    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)
    model = GlowTTS(config, ap, tokenizer=tokenizer, speaker_manager=None)
    
    # 5. Initialize the Trainer
    # ------------------------------------------------------------------
    # FIX: RESUMING TRAINING from Checkpoint 25609
    # We added 'restore_path' to TrainerArgs to load your previous progress.
    # ------------------------------------------------------------------
    logger = TensorboardLogger(log_dir=OUTPUT_PATH, model_name="glow_tts")

    # 👇 THIS IS THE PATH FROM YOUR SCREENSHOT
    CHECKPOINT_PATH = os.path.join(BASE_DIR, "proof_logs", "run_2026-02-12_01-24-57", "run-February-12-2026_01+24AM-0000000", "checkpoint_25609.pth")

    trainer = Trainer(
        TrainerArgs(restore_path=CHECKPOINT_PATH), # <--- This tells it to RESUME!
        config,
        OUTPUT_PATH,
        model=model,
        train_samples=train_samples, 
        eval_samples=eval_samples,   
        dashboard_logger=logger, 
    )
    
    print("\n" + "█"*60)
    print("      🚂  TRAINING ENGINE READY TO START      ")
    print("█"*60)
    print(f"📂 PROOF LOCATION:  {OUTPUT_PATH}")
    print("🛑 HOW TO STOP:     Press Ctrl+C (It will save safely)")
    print("█"*60 + "\n")
    
    try:
        trainer.fit()
    except KeyboardInterrupt:
        print("\n\n🛑 TRAINING PAUSED BY USER.")
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()