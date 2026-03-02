"""
Simple config loader that reads from YAML or JSON files.
"""

from pathlib import Path
from typing import Dict, Any
import json

try:
    import yaml
except ImportError:
    yaml = None


class Config:
    """Simple configuration class."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize config from a dictionary."""
        self._config = config_dict
        
        # Convert paths to Path objects
        if "paths" in config_dict:
            paths = config_dict["paths"]
            if "data_root" in paths:
                paths["data_root"] = Path(paths["data_root"])
            if "out_root" in paths:
                paths["out_root"] = Path(paths["out_root"])
            if "project_root" in paths:
                paths["project_root"] = Path(paths["project_root"]) if paths["project_root"] != "." else Path.cwd()
            if "lfw_dir" in paths:
                paths["lfw_dir"] = Path(paths["lfw_dir"])
            if "images_dir" in paths:
                paths["images_dir"] = Path(paths["images_dir"])
            if "manifests_dir" in paths:
                paths["manifests_dir"] = Path(paths["manifests_dir"])
            if "pairs_dir" in paths:
                paths["pairs_dir"] = Path(paths["pairs_dir"])
            if "similarity_score_dir" in paths:
                paths["similarity_score_dir"] = Path(paths["similarity_score_dir"])
        
        # Convert random values to int
        if "random" in config_dict:
            random = config_dict["random"]
            if "seed" in random:
                random["seed"] = int(random["seed"])
            if "pair_positive_offset" in random:
                random["pair_positive_offset"] = int(random["pair_positive_offset"])
            if "pair_negative_offset" in random:
                random["pair_negative_offset"] = int(random["pair_negative_offset"])
            if "pair_shuffle_offset" in random:
                random["pair_shuffle_offset"] = int(random["pair_shuffle_offset"])
        
        # Convert split ratios to float
        if "split" in config_dict:
            split = config_dict["split"]
            if "train_ratio" in split:
                split["train_ratio"] = float(split["train_ratio"])
            if "val_ratio" in split:
                split["val_ratio"] = float(split["val_ratio"])
            if "test_ratio" in split:
                split["test_ratio"] = float(split["test_ratio"])
        
        # Convert image settings
        if "image" in config_dict:
            image = config_dict["image"]
            if "size" in image:
                size = image["size"]
                if isinstance(size, list):
                    image["size"] = tuple(size)
            if "quality" in image:
                image["quality"] = int(image["quality"])
        
        # Convert pairs values to int
        if "pairs" in config_dict:
            pairs = config_dict["pairs"]
            if "num_positive_pairs" in pairs:
                pairs["num_positive_pairs"] = int(pairs["num_positive_pairs"])
            if "num_negative_pairs" in pairs:
                pairs["num_negative_pairs"] = int(pairs["num_negative_pairs"])
            if "max_attempts_multiplier" in pairs:
                pairs["max_attempts_multiplier"] = int(pairs["max_attempts_multiplier"])
        
        # Convert embedding values
        if "embedding" in config_dict:
            embedding = config_dict["embedding"]
            if "dimension" in embedding:
                embedding["dimension"] = int(embedding["dimension"])
            if "normalization_value" in embedding:
                embedding["normalization_value"] = float(embedding["normalization_value"])
        
        # Convert similarity epsilon to float
        if "similarity" in config_dict:
            similarity = config_dict["similarity"]
            if "epsilon" in similarity:
                similarity["epsilon"] = float(similarity["epsilon"])
        
        # Convert benchmark values
        if "benchmark" in config_dict:
            benchmark = config_dict["benchmark"]
            if "num_pairs" in benchmark:
                benchmark["num_pairs"] = int(benchmark["num_pairs"])
            if "tolerance" in benchmark:
                benchmark["tolerance"] = float(benchmark["tolerance"])
            if "benchmark_dimension" in benchmark:
                benchmark["benchmark_dimension"] = int(benchmark["benchmark_dimension"])
    
    def __getattr__(self, name: str):
        """Access nested config sections."""
        if name in self._config:
            section = self._config[name]
            # Return a simple object that allows attribute access
            return type("Section", (), section)()
        raise AttributeError(f"Config has no section '{name}'")
    
    @classmethod
    def from_file(cls, config_path: Path | str) -> "Config":
        """Load configuration from a YAML or JSON file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with config_path.open("r", encoding="utf-8") as f:
            if config_path.suffix.lower() in (".yaml", ".yml"):
                if yaml is None:
                    raise ImportError(
                        "PyYAML is required to load YAML config files. "
                        "Install it with: pip install pyyaml"
                    )
                config_dict = yaml.safe_load(f)
            elif config_path.suffix.lower() == ".json":
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        return cls(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create a Config instance from a dictionary."""
        return cls(config_dict)
