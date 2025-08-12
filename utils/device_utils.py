import torch
import logging
import psutil
import os
from typing import Dict, List, Optional, Tuple, Union
import subprocess
import platform


def setup_device(logger: Optional[logging.Logger] = None) -> torch.device:
    if logger is None:
        logger = logging.getLogger("DeviceUtils")
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device = torch.device('cuda')
        logger.info(f"CUDA available with {device_count} GPU(s)")
        logger.info(f"Primary GPU: {torch.cuda.get_device_name(0)}")
        if hasattr(torch.cuda, 'set_memory_fraction'):
            torch.cuda.set_memory_fraction(0.95)
        torch.backends.cudnn.benchmark = True
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("MPS (Apple Silicon) device available")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device")
        torch.set_num_threads(min(8, os.cpu_count()))
    return device


def get_memory_info(device: torch.device) -> Dict[str, Union[float, str]]:
    memory_info = {}
    if device.type == 'cuda':
        gpu_memory = torch.cuda.get_device_properties(device.index).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        cached_memory = torch.cuda.memory_reserved(device)
        memory_info.update({
            'device_type': 'CUDA',
            'device_name': torch.cuda.get_device_name(device),
            'total_memory_gb': gpu_memory / (1024**3),
            'allocated_memory_gb': allocated_memory / (1024**3),
            'cached_memory_gb': cached_memory / (1024**3),
            'free_memory_gb': (gpu_memory - cached_memory) / (1024**3),
            'memory_utilization': allocated_memory / gpu_memory * 100
        })
    elif device.type == 'mps':
        memory_info.update({
            'device_type': 'MPS',
            'device_name': 'Apple Silicon GPU'
        })
    else:
        ram = psutil.virtual_memory()
        memory_info.update({
            'device_type': 'CPU',
            'device_name': platform.processor(),
            'total_memory_gb': ram.total / (1024**3),
            'available_memory_gb': ram.available / (1024**3),
            'memory_utilization': ram.percent
        })
    return memory_info


def optimize_memory_usage(
    device: torch.device,
    model: torch.nn.Module,
    logger: Optional[logging.Logger] = None
) -> Dict[str, str]:
    if logger is None:
        logger = logging.getLogger("DeviceUtils")
    optimizations = {}
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        optimizations['cuda_cache_cleared'] = 'Yes'
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
            optimizations['flash_attention'] = 'Enabled'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        optimizations['memory_allocation'] = 'Optimized'
        logger.info("CUDA memory optimizations applied")
    elif device.type == 'cpu':
        torch.set_num_threads(min(8, os.cpu_count()))
        optimizations['cpu_threads'] = str(torch.get_num_threads())
        torch.set_grad_enabled(True)
        optimizations['cpu_optimization'] = 'Enabled'
        logger.info("CPU optimizations applied")
    return optimizations


def monitor_device_usage(device: torch.device) -> Dict[str, float]:
    usage_stats = {}
    if device.type == 'cuda':
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines and device.index < len(lines):
                    gpu_util, mem_used, mem_total = lines[device.index].split(', ')
                    usage_stats.update({
                        'gpu_utilization': float(gpu_util),
                        'memory_used_mb': float(mem_used),
                        'memory_total_mb': float(mem_total),
                        'memory_utilization': float(mem_used) / float(mem_total) * 100
                    })
        except:
            usage_stats.update({
                'memory_allocated_gb': torch.cuda.memory_allocated(device) / (1024**3),
                'memory_reserved_gb': torch.cuda.memory_reserved(device) / (1024**3)
            })
    elif device.type == 'cpu':
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        usage_stats.update({
            'cpu_utilization': cpu_percent,
            'memory_utilization': memory.percent,
            'memory_used_gb': (memory.total - memory.available) / (1024**3)
        })
    return usage_stats


def cleanup_device_memory(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class DeviceMonitor:
    def __init__(self, device: torch.device, log_interval: int = 100):
        self.device = device
        self.log_interval = log_interval
        self.step_count = 0
        self.logger = logging.getLogger("DeviceMonitor")
    
    def step(self):
        self.step_count += 1
        if self.step_count % self.log_interval == 0:
            usage_stats = monitor_device_usage(self.device)
            if self.device.type == 'cuda':
                mem_util = usage_stats.get('memory_utilization', 0)
                if mem_util > 90:
                    self.logger.warning(f"High GPU memory usage: {mem_util:.1f}%")
                    cleanup_device_memory(self.device)
            elif self.device.type == 'cpu':
                mem_util = usage_stats.get('memory_utilization', 0)
                if mem_util > 85:
                    self.logger.warning(f"High RAM usage: {mem_util:.1f}%")
