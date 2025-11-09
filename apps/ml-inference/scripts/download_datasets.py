#!/usr/bin/env python3
"""
Dataset Download Script for AgriMind ML Inference
Downloads all required datasets for plant disease detection with progress tracking.
"""

import os
import subprocess
import sys
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional
import shutil

class ProgressTracker:
    """Track download progress and display statistics."""
    
    def __init__(self):
        self.datasets_info = {}
        self.current_dataset = None
        self.start_time = None
        self.stop_monitoring = False
        
    def format_size(self, size_bytes: int) -> str:
        """Format size in bytes to human readable format."""
        if size_bytes == 0:
            return "0B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        import math
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"
    
    def get_directory_size(self, path: Path) -> int:
        """Get total size of directory in bytes."""
        total = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if os.path.exists(fp):
                        total += os.path.getsize(fp)
        except (OSError, FileNotFoundError):
            return 0
        return total
    
    def monitor_download(self, dataset_name: str, target_dir: Path):
        """Monitor download progress in real-time."""
        self.current_dataset = dataset_name
        self.stop_monitoring = False
        
        print(f"📊 Monitoring download progress for {dataset_name}...")
        
        while not self.stop_monitoring:
            if target_dir.exists():
                current_size = self.get_directory_size(target_dir)
                if current_size > 0:
                    formatted_size = self.format_size(current_size)
                    print(f"   📦 Downloaded: {formatted_size}", end='\r')
            
            time.sleep(1)
    
    def start_monitoring(self, dataset_name: str, target_dir: Path):
        """Start monitoring in a separate thread."""
        monitor_thread = threading.Thread(
            target=self.monitor_download,
            args=(dataset_name, target_dir)
        )
        monitor_thread.daemon = True
        monitor_thread.start()
        return monitor_thread
    
    def stop_monitoring_func(self):
        """Stop the monitoring process."""
        self.stop_monitoring = True

def ensure_kaggle_auth():
    """Ensure Kaggle API credentials are available."""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_file = kaggle_dir / 'kaggle.json'
    
    if not kaggle_file.exists():
        print("❌ Kaggle credentials not found. Please set up Kaggle API:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Create API token and download kaggle.json")
        print("3. Place it at ~/.kaggle/kaggle.json")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        sys.exit(1)
    
    # Check permissions
    file_stat = kaggle_file.stat()
    if file_stat.st_mode & 0o077:
        print("⚠️  Fixing Kaggle credentials permissions...")
        kaggle_file.chmod(0o600)
    
    print("✅ Kaggle API credentials found and configured")

def get_dataset_info(dataset_id: str) -> Dict:
    """Get dataset information from Kaggle API."""
    try:
        cmd = f"kaggle datasets list -s {dataset_id} --csv"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:
            # Parse CSV output
            header = lines[0].split(',')
            data = lines[1].split(',')
            
            info = {}
            for i, field in enumerate(header):
                if i < len(data):
                    info[field] = data[i]
            
            return info
    except Exception as e:
        print(f"⚠️  Could not fetch dataset info for {dataset_id}: {e}")
    
    return {}

def run_command_with_progress(command: str, description: str, dataset_name: str, target_dir: Path, progress_tracker: ProgressTracker):
    """Run a command with progress monitoring."""
    print(f"\n🔄 {description}")
    print(f"📍 Target directory: {target_dir}")
    
    # Start progress monitoring
    monitor_thread = progress_tracker.start_monitoring(dataset_name, target_dir)
    
    start_time = time.time()
    
    try:
        # Run the download command
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            universal_newlines=True
        )
        
        # Monitor process output
        stdout_lines = []
        stderr_lines = []
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                stdout_lines.append(output.strip())
                # Show relevant download progress from kaggle CLI
                if "Downloading" in output or "%" in output:
                    print(f"   {output.strip()}")
        
        # Get any remaining output
        stdout, stderr = process.communicate()
        if stdout:
            stdout_lines.extend(stdout.strip().split('\n'))
        if stderr:
            stderr_lines.extend(stderr.strip().split('\n'))
        
        # Stop monitoring
        progress_tracker.stop_monitoring_func()
        
        elapsed_time = time.time() - start_time
        
        if process.returncode == 0:
            # Get final size
            if target_dir.exists():
                final_size = progress_tracker.get_directory_size(target_dir)
                formatted_size = progress_tracker.format_size(final_size)
                
                print(f"\n✅ {description} completed successfully!")
                print(f"   📦 Final size: {formatted_size}")
                print(f"   ⏱️  Time taken: {elapsed_time:.1f} seconds")
                
                # Count files
                file_count = sum(1 for _ in target_dir.rglob('*') if _.is_file())
                print(f"   📄 Files downloaded: {file_count:,}")
            
            return True
        else:
            print(f"\n❌ {description} failed!")
            if stderr_lines:
                print("Error details:")
                for line in stderr_lines:
                    if line.strip():
                        print(f"   {line}")
            return False
            
    except Exception as e:
        progress_tracker.stop_monitoring_func()
        print(f"\n❌ {description} failed with exception: {e}")
        return False

def download_datasets():
    """Download all required datasets with enhanced progress tracking."""
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Initialize progress tracker
    progress_tracker = ProgressTracker()
    
    datasets = [
        {
            "id": "emmarex/plantdisease",
            "command": "kaggle datasets download -d emmarex/plantdisease -p data/plantvillage --unzip",
            "description": "PlantVillage Dataset",
            "target_dir": data_dir / "plantvillage",
            "estimated_size": "870 MB"
        },
        {
            "id": "pratik1120/plantdoc-dataset",
            "command": "kaggle datasets download -d pratik1120/plantdoc-dataset -p data/plantdoc --unzip",
            "description": "PlantDoc Dataset",
            "target_dir": data_dir / "plantdoc",
            "estimated_size": "416 MB"
        },
        {
            "id": "nafishamoin/bangladeshi-crops-disease-dataset",
            "command": "kaggle datasets download -d nafishamoin/bangladeshi-crops-disease-dataset -p data/bangla_crops --unzip",
            "description": "Bangladeshi Crops Disease Dataset",
            "target_dir": data_dir / "bangla_crops",
            "estimated_size": "125 MB"
        },
        {
            "id": "vbookshelf/rice-leaf-diseases",
            "command": "kaggle datasets download -d vbookshelf/rice-leaf-diseases -p data/rice_leaf --unzip",
            "description": "Rice Leaf Diseases Dataset",
            "target_dir": data_dir / "rice_leaf",
            "estimated_size": "48 MB"
        }
    ]
    
    print(f"📊 Preparing to download {len(datasets)} datasets")
    print(f"💾 Total estimated size: ~1.46 GB")
    print("=" * 60)
    
    success_count = 0
    total_downloaded_size = 0
    
    for i, dataset in enumerate(datasets, 1):
        print(f"\n📦 Dataset {i}/{len(datasets)}: {dataset['description']}")
        print(f"🔗 Kaggle ID: {dataset['id']}")
        print(f"📏 Estimated size: {dataset['estimated_size']}")
        
        # Check if dataset already exists
        if dataset['target_dir'].exists() and any(dataset['target_dir'].iterdir()):
            existing_size = progress_tracker.get_directory_size(dataset['target_dir'])
            formatted_size = progress_tracker.format_size(existing_size)
            print(f"⚠️  Dataset already exists ({formatted_size}). Skipping...")
            success_count += 1
            total_downloaded_size += existing_size
            continue
        
        # Get dataset info
        dataset_info = get_dataset_info(dataset['id'])
        if dataset_info:
            print(f"📊 Dataset info: {dataset_info.get('title', 'N/A')}")
        
        # Download with progress tracking
        if run_command_with_progress(
            dataset["command"], 
            f"Downloading {dataset['description']}", 
            dataset['description'],
            dataset['target_dir'],
            progress_tracker
        ):
            success_count += 1
            if dataset['target_dir'].exists():
                size = progress_tracker.get_directory_size(dataset['target_dir'])
                total_downloaded_size += size
        
        print("-" * 40)
    
    # Summary
    print(f"\n🎉 DOWNLOAD SUMMARY")
    print("=" * 40)
    print(f"✅ Successfully downloaded: {success_count}/{len(datasets)} datasets")
    print(f"📦 Total size downloaded: {progress_tracker.format_size(total_downloaded_size)}")
    
    if success_count < len(datasets):
        print(f"⚠️  Failed downloads: {len(datasets) - success_count}")
        print("💡 You can re-run this script to retry failed downloads")
    
    return success_count
    
    # Create placeholder for Roboflow data
    roboflow_dir = data_dir / "indian_trees"
    roboflow_dir.mkdir(exist_ok=True)
    
    roboflow_readme = roboflow_dir / "README.md"
    roboflow_readme.write_text("""# Indian Trees Dataset from Roboflow

This directory is for the Indian Trees dataset from Roboflow.

## Setup Instructions:
1. Go to your Roboflow project
2. Export dataset in YOLOv8 format
3. Download and extract to this directory
4. Update the dataset configuration in `src/config.py`

## Expected Structure:
```
indian_trees/
├── train/
├── valid/
├── test/
└── data.yaml
```
""")
    
    print("📁 Created placeholder for Roboflow Indian Trees dataset")
    print("   Please follow instructions in data/indian_trees/README.md")

def create_roboflow_placeholder():
    """Create placeholder structure for Roboflow dataset."""
    data_dir = Path("data")
    roboflow_dir = data_dir / "indian_trees"
    roboflow_dir.mkdir(exist_ok=True)
    
    roboflow_readme = roboflow_dir / "README.md"
    roboflow_readme.write_text("""# Indian Trees Dataset from Roboflow

This directory is for the Indian Trees dataset from Roboflow.

## Setup Instructions:
1. Go to your Roboflow project
2. Export dataset in YOLOv8 format
3. Download and extract to this directory
4. Update the dataset configuration in `src/config.py`

## Expected Structure:
```
indian_trees/
├── train/
├── valid/
├── test/
└── data.yaml
```

## Import Instructions:
```python
# In your Roboflow project
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("YOUR_WORKSPACE").project("YOUR_PROJECT")
dataset = project.version(VERSION_NUMBER).download("yolov8", location="./data/indian_trees")
```
""")
    
    print("📁 Created placeholder for Roboflow Indian Trees dataset")
    print(f"   📍 Location: {roboflow_dir}")
    print("   📖 Instructions: data/indian_trees/README.md")

def main():
    """Main function with enhanced progress tracking."""
    print("🌱 AgriMind Dataset Downloader v2.0")
    print("🚀 Enhanced with Progress Tracking & Size Monitoring")
    print("=" * 60)
    
    # Check system requirements
    print("🔍 Checking system requirements...")
    
    # Check available disk space
    try:
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)
        print(f"💾 Available disk space: {free_gb:.1f} GB")
        
        if free_gb < 2.0:
            print("⚠️  Warning: Less than 2GB free space available!")
            print("   Datasets require approximately 1.5GB of space")
    except:
        print("⚠️  Could not check disk space")
    
    # Ensure Kaggle authentication
    ensure_kaggle_auth()
    
    # Start downloads
    start_time = time.time()
    success_count = download_datasets()
    
    # Create Roboflow placeholder
    create_roboflow_placeholder()
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n🎉 PROCESS COMPLETED!")
    print("=" * 40)
    print(f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"📊 Success rate: {success_count}/4 Kaggle datasets")
    
    if success_count == 4:
        print("🎊 All datasets downloaded successfully!")
    else:
        print("⚠️  Some downloads may have failed. Check the logs above.")
    
    print(f"\n📋 NEXT STEPS:")
    print("1. 🔗 Set up Roboflow dataset (see data/indian_trees/README.md)")
    print("2. 🔄 Run data preprocessing: python src/data_preprocessing.py")
    print("3. 🚀 Start training: python src/train.py")
    print("4. 🌐 Or run full pipeline: python src/main.py pipeline")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
        print("💡 You can re-run this script to resume downloads")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("💡 Please check your internet connection and Kaggle credentials")
        sys.exit(1)
