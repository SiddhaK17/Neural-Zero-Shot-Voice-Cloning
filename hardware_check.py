import torch
import sys
import os
import psutil
import shutil  # <--- MOVING THIS TO TOP FIXED THE BUG

def print_status():
    print("\n" + "="*50)
    print("      🔍 PRE-FLIGHT SYSTEM DIAGNOSTICS      ")
    print("="*50)

    # 1. GPU Check
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        # Fix: Calculate VRAM safely
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU DETECTED: {gpu_name}")
        print(f"ℹ️  VRAM Available: {vram:.2f} GB")
        
        if vram < 6.0:
            print("⚠️  WARNING: VRAM is below 6GB. Training will be slow.")
        else:
            print("🚀 STATUS: GPU is ready for High-Performance Training.")
    else:
        print("❌ CRITICAL ERROR: No NVIDIA GPU Detected.")
        print("   The training will burn out your CPU. ABORTING.")
        sys.exit(1)

    # 2. Power Check
    battery = psutil.sensors_battery()
    if battery:
        plugged = battery.power_plugged
        if not plugged:
            print("❌ CRITICAL WARNING: Laptop is running on BATTERY.")
            print("   Please plug in your charger to prevent shutdown.")
            sys.exit(1)
        else:
            print("✅ POWER: AC Adapter Connected.")
    else:
        print("✅ POWER: Desktop / No Battery Detected.")

    # 3. Storage Check
    free_space = shutil.disk_usage(os.getcwd()).free / 1e9
    print(f"💾 STORAGE: {free_space:.2f} GB Free.")
    if free_space < 10:
        print("⚠️  WARNING: Low Disk Space. You need at least 10GB.")
    
    print("="*50 + "\n")
    print("✅ SYSTEM CHECK PASSED. You may proceed to train.py")

if __name__ == "__main__":
    print_status()