import os
import yaml
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "frame": {
        "implementation": "pandas",
        "implementations": {
            "pandas": {},
            "cudf": {
                "device_id": 0,
                "memory_pool_size": "1GB"
            },
            "mnmg": {
                "devices": [0],
                "partition_strategy": "hash",
                "communication_backend": "nccl"
            },
            "scallop": {
                "mode": "exact"
            }
        }
    }
}

class Config:
    """Configuration manager for the Datalog engine."""
    
    _instance = None
    _config_dict = None
    _config_file = None
    
    @classmethod
    def get_instance(cls) -> 'Config':
        """Get the singleton instance of Config."""
        if cls._instance is None:
            cls._instance = Config()
        return cls._instance
    
    def __init__(self):
        """Initialize with default configuration."""
        if Config._instance is not None:
            raise RuntimeError("Config is a singleton. Use Config.get_instance() instead.")
        self._config_dict = DEFAULT_CONFIG.copy()
        
    def load_from_file(self, config_file: str) -> None:
        """Load configuration from a YAML file."""
        if not os.path.exists(config_file):
            logger.warning(f"Config file {config_file} not found. Using default configuration.")
            return
            
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                
            if not config:
                logger.warning("Empty config file. Using default configuration.")
                return
                
            # Update configuration, maintaining defaults for missing values
            self._update_dict_recursive(self._config_dict, config)
            self._config_file = config_file
            logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logger.error(f"Error loading config file {config_file}: {e}")
            
    def _update_dict_recursive(self, target: Dict, source: Dict) -> None:
        """Recursively update a dictionary, preserving keys not in source."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._update_dict_recursive(target[key], value)
            else:
                target[key] = value
    
    def get(self, path: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation path."""
        parts = path.split('.')
        current = self._config_dict
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
                
        return current
    
    def set(self, path: str, value: Any) -> None:
        """Set configuration value by dot-notation path."""
        parts = path.split('.')
        current = self._config_dict
        
        # Navigate to the parent of the target
        for i, part in enumerate(parts[:-1]):
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
            
        # Set the value
        current[parts[-1]] = value
        
    def get_frame_implementation(self) -> str:
        """Get the configured Frame implementation."""
        return self.get('frame.implementation', 'pandas')
    
    def get_implementation_config(self, implementation: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration for a specific Frame implementation."""
        if implementation is None:
            implementation = self.get_frame_implementation()
            
        config_path = f'frame.implementations.{implementation}'
        return self.get(config_path, {})
        
    def save(self, config_file: Optional[str] = None) -> None:
        """Save current configuration to a YAML file."""
        file_path = config_file or self._config_file
        
        if not file_path:
            logger.warning("No config file specified for saving.")
            return
            
        try:
            with open(file_path, 'w') as f:
                yaml.dump(self._config_dict, f, default_flow_style=False)
            logger.info(f"Saved configuration to {file_path}")
        except Exception as e:
            logger.error(f"Error saving config to {file_path}: {e}")
            
# Singleton instance
config = Config.get_instance()
