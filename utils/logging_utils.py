import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import torch
import psutil
import platform


def setup_comprehensive_logging(
    output_dir: str,
    log_level: str = "INFO",
    experiment_name: Optional[str] = None
) -> logging.Logger:
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    if experiment_name is None:
        experiment_name = f"radixpert_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = logging.getLogger("Radixpert")
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        fmt='%(levelname)s: %(message)s'
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    detailed_log_file = log_dir / f"{experiment_name}_detailed.log"
    file_handler = logging.handlers.RotatingFileHandler(
        detailed_log_file,
        maxBytes=50*1024*1024,
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    error_log_file = log_dir / f"{experiment_name}_errors.log"
    error_handler = logging.FileHandler(error_log_file)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    logger.addHandler(error_handler)
    logger.info(f"Logging system initialized - logs saved to {log_dir}")
    return logger


def log_system_info(logger: logging.Logger):
    logger.info("SYSTEM INFORMATION:")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Python: {sys.version}")
    logger.info(f"  PyTorch: {torch.__version__}")
    logger.info(f"  CPU: {platform.processor()}")
    logger.info(f"  CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    memory = psutil.virtual_memory()
    logger.info(f"  RAM: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available")
    if torch.cuda.is_available():
        logger.info(f"  CUDA: {torch.version.cuda}")
        logger.info(f"  GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger.info(f"    GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        logger.info("  GPU: None (CUDA not available)")


def log_model_info(model: torch.nn.Module, logger: logging.Logger):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("MODEL INFORMATION:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    if hasattr(model, 'get_trainable_parameters'):
        component_params = model.get_trainable_parameters()
        logger.info("  Parameter breakdown:")
        for component, count in component_params.items():
            if isinstance(count, dict):
                logger.info(f"    {component}:")
                for subcomp, subcount in count.items():
                    logger.info(f"      {subcomp}: {subcount:,}")
            else:
                logger.info(f"    {component}: {count:,}")
