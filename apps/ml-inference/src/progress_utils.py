"""
Enhanced progress utilities for AgriMind ML operations.
"""

import time
import sys
from typing import Optional, Callable

class ProgressBar:
    """A simple progress bar for console applications."""
    
    def __init__(self, total: int, description: str = "", width: int = 50):
        self.total = total
        self.current = 0
        self.description = description
        self.width = width
        self.start_time = time.time()
    
    def update(self, amount: int = 1, description: Optional[str] = None):
        """Update the progress bar."""
        self.current = min(self.current + amount, self.total)
        if description:
            self.description = description
        self._render()
    
    def set_progress(self, current: int, description: Optional[str] = None):
        """Set absolute progress."""
        self.current = min(max(current, 0), self.total)
        if description:
            self.description = description
        self._render()
    
    def _render(self):
        """Render the progress bar."""
        if self.total == 0:
            percent = 100
        else:
            percent = (self.current / self.total) * 100
        
        filled_width = int(self.width * self.current / max(self.total, 1))
        bar = '█' * filled_width + '░' * (self.width - filled_width)
        
        elapsed_time = time.time() - self.start_time
        if self.current > 0 and elapsed_time > 0:
            rate = self.current / elapsed_time
            eta = (self.total - self.current) / rate if rate > 0 else 0
            time_info = f" | ETA: {eta:.0f}s"
        else:
            time_info = ""
        
        progress_line = f"\r{self.description} |{bar}| {self.current}/{self.total} ({percent:.1f}%){time_info}"
        
        # Ensure we don't exceed terminal width
        if len(progress_line) > 100:
            progress_line = progress_line[:97] + "..."
        
        print(progress_line, end='', flush=True)
    
    def finish(self, message: str = "Complete!"):
        """Finish the progress bar."""
        self.current = self.total
        self._render()
        print(f"\n✅ {message}")

class SpinnerProgress:
    """A spinner for indeterminate progress."""
    
    def __init__(self, description: str = "Processing..."):
        self.description = description
        self.spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.current_char = 0
        self.is_spinning = False
    
    def start(self):
        """Start the spinner."""
        self.is_spinning = True
        self._spin()
    
    def stop(self, message: str = "Done!"):
        """Stop the spinner."""
        self.is_spinning = False
        print(f"\r✅ {message}                    ")
    
    def _spin(self):
        """Render spinner animation."""
        if self.is_spinning:
            char = self.spinner_chars[self.current_char]
            print(f"\r{char} {self.description}...", end='', flush=True)
            self.current_char = (self.current_char + 1) % len(self.spinner_chars)

def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def format_bytes(bytes_value: int) -> str:
    """Format bytes to human readable format."""
    if bytes_value == 0:
        return "0B"
    
    sizes = ['B', 'KB', 'MB', 'GB', 'TB']
    import math
    i = int(math.floor(math.log(bytes_value, 1024)))
    p = math.pow(1024, i)
    s = round(bytes_value / p, 2)
    return f"{s} {sizes[i]}"

class MultiStageProgress:
    """Track progress across multiple stages/phases."""
    
    def __init__(self, stages: list, total_items: Optional[int] = None):
        self.stages = stages
        self.current_stage = 0
        self.current_stage_progress = 0
        self.stage_totals = {}
        self.total_items = total_items or sum(self.stage_totals.values()) if self.stage_totals else len(stages)
        self.start_time = time.time()
        
        print(f"🚀 Starting {len(stages)} stage process...")
        for i, stage in enumerate(stages):
            print(f"   Stage {i+1}: {stage}")
        print()
    
    def start_stage(self, stage_index: int, stage_total: Optional[int] = None):
        """Start a new stage."""
        if stage_index < len(self.stages):
            self.current_stage = stage_index
            self.current_stage_progress = 0
            
            if stage_total:
                self.stage_totals[stage_index] = stage_total
            
            stage_name = self.stages[stage_index]
            print(f"\n📍 Stage {stage_index + 1}/{len(self.stages)}: {stage_name}")
            print("-" * 50)
    
    def update_stage_progress(self, amount: int = 1, description: str = ""):
        """Update progress for current stage."""
        self.current_stage_progress += amount
        
        # Calculate overall progress
        completed_stages = self.current_stage
        total_progress = completed_stages
        
        if self.current_stage < len(self.stages):
            stage_total = self.stage_totals.get(self.current_stage, 1)
            stage_progress = min(self.current_stage_progress / stage_total, 1.0)
            total_progress += stage_progress
        
        overall_percent = (total_progress / len(self.stages)) * 100
        
        elapsed_time = time.time() - self.start_time
        
        stage_name = self.stages[self.current_stage] if self.current_stage < len(self.stages) else "Complete"
        
        print(f"   📊 {stage_name}: {description}")
        print(f"   🔄 Overall Progress: {overall_percent:.1f}% | Time: {format_time(elapsed_time)}")
    
    def complete_stage(self):
        """Mark current stage as complete."""
        stage_name = self.stages[self.current_stage] if self.current_stage < len(self.stages) else "Stage"
        print(f"   ✅ {stage_name} completed!")
    
    def complete(self):
        """Mark entire process as complete."""
        total_time = time.time() - self.start_time
        print(f"\n🎉 All stages completed in {format_time(total_time)}!")

# Example usage functions for common ML operations
def track_dataset_download(datasets: list) -> MultiStageProgress:
    """Create progress tracker for dataset downloads."""
    stages = [
        "Checking Kaggle credentials",
        "Downloading datasets",
        "Extracting and organizing",
        "Validating downloads"
    ]
    
    tracker = MultiStageProgress(stages)
    tracker.start_stage(0)
    return tracker

def track_training_progress(epochs: int, batches_per_epoch: int) -> dict:
    """Create progress tracking for model training."""
    return {
        'epoch_progress': ProgressBar(epochs, "Training Epochs"),
        'batch_progress': ProgressBar(batches_per_epoch, "Batch Progress"),
        'start_time': time.time()
    }

def show_system_info():
    """Display system information relevant to ML operations."""
    import platform
    import psutil
    
    print("💻 SYSTEM INFORMATION")
    print("=" * 40)
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")
    
    # Memory info
    memory = psutil.virtual_memory()
    print(f"RAM: {format_bytes(memory.total)} total, {format_bytes(memory.available)} available")
    
    # Disk info
    disk = psutil.disk_usage('.')
    print(f"Disk Space: {format_bytes(disk.free)} free of {format_bytes(disk.total)}")
    
    # Check for GPU (if available)
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"GPU: {gpu_count} CUDA device(s) available")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory
                print(f"  GPU {i}: {gpu_name} ({format_bytes(memory_total)})")
        else:
            print("GPU: No CUDA devices available")
    except ImportError:
        print("GPU: PyTorch not installed, cannot check GPU status")
    
    print()

if __name__ == "__main__":
    # Demo of progress utilities
    print("🔧 AgriMind Progress Utilities Demo")
    print("=" * 40)
    
    # Demo progress bar
    print("\n1. Progress Bar Demo:")
    pbar = ProgressBar(100, "Demo Progress")
    for i in range(101):
        pbar.update(1, f"Processing item {i}")
        time.sleep(0.01)  # Simulate work
    pbar.finish("Demo completed!")
    
    # Demo multi-stage
    print("\n2. Multi-Stage Progress Demo:")
    stages = ["Stage 1", "Stage 2", "Stage 3"]
    tracker = MultiStageProgress(stages)
    
    for stage_idx in range(len(stages)):
        tracker.start_stage(stage_idx, 5)
        for item in range(5):
            tracker.update_stage_progress(1, f"Item {item+1}")
            time.sleep(0.1)
        tracker.complete_stage()
    
    tracker.complete()
