"""
Professional Error Handling and Logging System for iLLuMinator 4.9B
Centralized error handling with comprehensive logging and recovery mechanisms
"""

import logging
import traceback
import sys
import os
import functools
from typing import Any, Callable, Optional
from pathlib import Path
import warnings

class ProfessionalLogger:
    """Professional logging setup with multiple handlers and formats"""
    
    def __init__(self, name: str = "illuminator", log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(name)
        self.setup_logging()
    
    def setup_logging(self):
        """Setup comprehensive logging configuration"""
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Set base level
        self.logger.setLevel(logging.DEBUG)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Console handler (INFO and above)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for all logs
        file_handler = logging.FileHandler(self.log_dir / "illuminator.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(file_handler)
        
        # Error file handler
        error_handler = logging.FileHandler(self.log_dir / "errors.log")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(error_handler)
        
        # Training-specific handler
        training_handler = logging.FileHandler(self.log_dir / "training.log")
        training_handler.setLevel(logging.INFO)
        training_handler.setFormatter(detailed_formatter)
        
        # Create a filter for training-related logs
        class TrainingFilter(logging.Filter):
            def filter(self, record):
                return any(keyword in record.getMessage().lower() 
                          for keyword in ['epoch', 'loss', 'training', 'batch', 'step'])
        
        training_handler.addFilter(TrainingFilter())
        self.logger.addHandler(training_handler)
    
    def get_logger(self):
        """Get the configured logger"""
        return self.logger

class ErrorHandler:
    """Professional error handling with graceful degradation"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or ProfessionalLogger().get_logger()
    
    def handle_import_error(self, module_name: str, fallback_action: Optional[Callable] = None):
        """Handle import errors with fallback options"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except ImportError as e:
                    self.logger.error(f"Failed to import {module_name}: {e}")
                    if fallback_action:
                        self.logger.info(f"Using fallback action for {module_name}")
                        return fallback_action(*args, **kwargs)
                    else:
                        self.logger.error(f"No fallback available for {module_name}")
                        raise
            return wrapper
        return decorator
    
    def handle_file_error(self, operation: str = "file operation"):
        """Handle file I/O errors gracefully"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except FileNotFoundError as e:
                    self.logger.error(f"File not found during {operation}: {e}")
                    # Try to create directory if it doesn't exist
                    if hasattr(e, 'filename') and e.filename:
                        parent_dir = Path(e.filename).parent
                        if not parent_dir.exists():
                            parent_dir.mkdir(parents=True, exist_ok=True)
                            self.logger.info(f"Created directory: {parent_dir}")
                            # Retry the operation
                            return func(*args, **kwargs)
                    raise
                except PermissionError as e:
                    self.logger.error(f"Permission denied during {operation}: {e}")
                    raise
                except OSError as e:
                    self.logger.error(f"OS error during {operation}: {e}")
                    raise
            return wrapper
        return decorator
    
    def handle_cuda_error(self, fallback_device: str = "cpu"):
        """Handle CUDA errors with fallback to CPU"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except RuntimeError as e:
                    if "CUDA" in str(e) or "out of memory" in str(e):
                        self.logger.error(f"CUDA error: {e}")
                        self.logger.info(f"Falling back to {fallback_device}")
                        # Try to modify device in kwargs if present
                        if 'device' in kwargs:
                            kwargs['device'] = fallback_device
                        return func(*args, **kwargs)
                    else:
                        raise
            return wrapper
        return decorator
    
    def safe_execute(self, func: Callable, *args, **kwargs) -> Any:
        """Safely execute a function with comprehensive error handling"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Error executing {func.__name__}: {e}")
            self.logger.debug(f"Full traceback:\n{traceback.format_exc()}")
            raise

def setup_professional_error_handling():
    """Setup global error handling and warnings"""
    # Setup logging
    logger = ProfessionalLogger().get_logger()
    
    # Handle uncaught exceptions
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
    
    sys.excepthook = handle_exception
    
    # Setup warning handling
    def handle_warning(message, category, filename, lineno, file=None, line=None):
        logger.warning(f"{category.__name__}: {message} ({filename}:{lineno})")
    
    warnings.showwarning = handle_warning
    
    logger.info("Professional error handling initialized")
    return logger

def safe_import(module_name: str, fallback=None, logger=None):
    """Safely import a module with fallback"""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        if '.' in module_name:
            # Handle dotted imports
            parts = module_name.split('.')
            module = __import__(module_name)
            for part in parts[1:]:
                module = getattr(module, part)
        else:
            module = __import__(module_name)
        
        logger.info(f"Successfully imported {module_name}")
        return module
    except ImportError as e:
        logger.warning(f"Failed to import {module_name}: {e}")
        if fallback is not None:
            logger.info(f"Using fallback for {module_name}")
            return fallback
        return None

def validate_environment():
    """Validate the environment and log important information"""
    logger = ProfessionalLogger().get_logger()
    
    logger.info("Environment Validation")
    logger.info("=" * 50)
    
    # Python version
    logger.info(f"Python version: {sys.version}")
    
    # PyTorch availability
    torch = safe_import('torch', logger=logger)
    if torch:
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")
        
        # MPS availability (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("Apple Silicon MPS: Available")
    
    # Check critical directories
    critical_dirs = ['checkpoints', 'logs', 'datasets']
    for dir_name in critical_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            logger.info(f"Directory '{dir_name}': Exists")
        else:
            dir_path.mkdir(exist_ok=True)
            logger.info(f"Directory '{dir_name}': Created")
    
    # Check disk space
    import shutil
    total, used, free = shutil.disk_usage(".")
    logger.info(f"Disk space: {free / 1e9:.1f}GB free / {total / 1e9:.1f}GB total")
    
    if free < 10e9:  # Less than 10GB
        logger.warning("Low disk space! Consider freeing up space before training.")
    
    logger.info("Environment validation completed")
    return True

# Convenience decorators for common use cases
def handle_training_errors(logger=None):
    """Decorator for training functions"""
    if logger is None:
        logger = ProfessionalLogger().get_logger()
    
    handler = ErrorHandler(logger)
    
    def decorator(func):
        @functools.wraps(func)
        @handler.handle_cuda_error()
        @handler.handle_file_error("training")
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Training error in {func.__name__}: {e}")
                logger.info("Attempting to save emergency checkpoint...")
                # Emergency save logic could go here
                raise
        return wrapper
    return decorator

def handle_data_errors(logger=None):
    """Decorator for data loading functions"""
    if logger is None:
        logger = ProfessionalLogger().get_logger()
    
    handler = ErrorHandler(logger)
    
    def decorator(func):
        @functools.wraps(func)
        @handler.handle_file_error("data loading")
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Data loading error in {func.__name__}: {e}")
                logger.info("Consider checking dataset integrity or network connection")
                raise
        return wrapper
    return decorator

def main():
    """Demo the professional error handling system"""
    print("Professional Error Handling System Demo")
    print("=" * 50)
    
    # Setup error handling
    logger = setup_professional_error_handling()
    
    # Validate environment
    validate_environment()
    
    # Test error handling
    @handle_training_errors(logger)
    def test_training_function():
        logger.info("Test training function executed successfully")
        return True
    
    @handle_data_errors(logger)
    def test_data_function():
        logger.info("Test data function executed successfully")
        return True
    
    # Execute test functions
    test_training_function()
    test_data_function()
    
    logger.info("Error handling demo completed successfully")

if __name__ == "__main__":
    main()