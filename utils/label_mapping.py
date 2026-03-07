"""
Label mapping utilities for cervical cancer classification.
Handles mapping between SIPaKMeD and Herlev datasets for binary and multi-class classification.
"""

from typing import Dict, List, Optional
from enum import Enum


class DatasetType(Enum):
    SIPAKMED = "sipakmed"
    HERLEV = "herlev"


class ClassificationMode(Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"


class LabelMapper:
    """
    Flexible label mapping system for cervical cancer classification.
    """
    
    # SIPaKMeD class definitions
    SIPAKMED_CLASSES = {
        0: "Superficial-Intermediate",
        1: "Parabasal", 
        2: "Koilocytotic",
        3: "Metaplastic",
        4: "Dyskeratotic"
    }
    
    # Herlev class definitions
    HERLEV_CLASSES = {
        0: "Normal superficial",
        1: "Normal intermediate", 
        2: "Normal columnar",
        3: "Mild dysplasia",
        4: "Moderate dysplasia",
        5: "Severe dysplasia",
        6: "Carcinoma in situ"
    }
    
    # Binary classification mapping
    BINARY_MAPPING = {
        # NORMAL classes
        "Superficial-Intermediate": 0,
        "Parabasal": 0,
        "Normal superficial": 0,
        "Normal intermediate": 0,
        "Normal columnar": 0,
        # ABNORMAL classes
        "Koilocytotic": 1,
        "Metaplastic": 1,
        "Dyskeratotic": 1,
        "Mild dysplasia": 1,
        "Moderate dysplasia": 1,
        "Severe dysplasia": 1,
        "Carcinoma in situ": 1
    }
    
    # Binary class names
    BINARY_CLASS_NAMES = {0: "NORMAL", 1: "ABNORMAL"}
    
    def __init__(self, mode: ClassificationMode = ClassificationMode.MULTICLASS):
        """
        Initialize label mapper.
        
        Args:
            mode: Classification mode (binary or multiclass)
        """
        self.mode = mode
        self._setup_mappings()
    
    def _setup_mappings(self):
        """Setup label mappings based on classification mode."""
        if self.mode == ClassificationMode.BINARY:
            self.num_classes = 2
            self.class_names = self.BINARY_CLASS_NAMES
        else:
            # For multiclass, we need to handle different datasets separately
            self.num_classes = None  # Will be set per dataset
            self.class_names = None
    
    def get_dataset_classes(self, dataset_type: DatasetType) -> Dict[int, str]:
        """Get class names for a specific dataset."""
        if dataset_type == DatasetType.SIPAKMED:
            return self.SIPAKMED_CLASSES
        elif dataset_type == DatasetType.HERLEV:
            return self.HERLEV_CLASSES
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    def map_label(self, label: int, dataset_type: DatasetType) -> int:
        """
        Map original label to target label based on classification mode.
        
        Args:
            label: Original label from dataset
            dataset_type: Type of dataset (SIPaKMeD or Herlev)
            
        Returns:
            Mapped label
        """
        if self.mode == ClassificationMode.MULTICLASS:
            # For multiclass, keep original labels but ensure they start from 0
            return label
        else:
            # For binary classification, map to NORMAL/ABNORMAL
            class_names = self.get_dataset_classes(dataset_type)
            class_name = class_names[label]
            return self.BINARY_MAPPING[class_name]
    
    def get_class_names(self, dataset_type: Optional[DatasetType] = None) -> List[str]:
        """
        Get class names for the current mode and dataset.
        
        Args:
            dataset_type: Required for multiclass mode
            
        Returns:
            List of class names
        """
        if self.mode == ClassificationMode.BINARY:
            return [self.BINARY_CLASS_NAMES[0], self.BINARY_CLASS_NAMES[1]]
        else:
            if dataset_type is None:
                raise ValueError("dataset_type required for multiclass mode")
            classes = self.get_dataset_classes(dataset_type)
            return [classes[i] for i in range(len(classes))]
    
    def get_num_classes(self, dataset_type: Optional[DatasetType] = None) -> int:
        """
        Get number of classes for current mode and dataset.
        
        Args:
            dataset_type: Required for multiclass mode
            
        Returns:
            Number of classes
        """
        if self.mode == ClassificationMode.BINARY:
            return 2
        else:
            if dataset_type is None:
                raise ValueError("dataset_type required for multiclass mode")
            classes = self.get_dataset_classes(dataset_type)
            return len(classes)
    
    def create_cross_dataset_mapping(self) -> Dict[str, Dict[str, int]]:
        """
        Create cross-dataset label mapping for evaluation.
        
        Returns:
            Dictionary mapping dataset-class pairs to binary labels
        """
        mapping = {}
        
        # SIPaKMeD mapping
        for label, class_name in self.SIPAKMED_CLASSES.items():
            key = f"sipakmed_{class_name}"
            mapping[key] = self.BINARY_MAPPING[class_name]
        
        # Herlev mapping  
        for label, class_name in self.HERLEV_CLASSES.items():
            key = f"herlev_{class_name}"
            mapping[key] = self.BINARY_MAPPING[class_name]
        
        return mapping


def get_label_mapper(mode: str = "multiclass") -> LabelMapper:
    """
    Factory function to create label mapper.
    
    Args:
        mode: Classification mode ("binary" or "multiclass")
        
    Returns:
        LabelMapper instance
    """
    if mode.lower() == "binary":
        return LabelMapper(ClassificationMode.BINARY)
    elif mode.lower() == "multiclass":
        return LabelMapper(ClassificationMode.MULTICLASS)
    else:
        raise ValueError(f"Unknown classification mode: {mode}")


# Utility functions for common operations
def is_normal_class(class_name: str) -> bool:
    """Check if a class name represents a normal cell."""
    normal_classes = [
        "Superficial-Intermediate",
        "Parabasal", 
        "Normal superficial",
        "Normal intermediate",
        "Normal columnar"
    ]
    return class_name in normal_classes


def get_binary_label(class_name: str) -> int:
    """Get binary label (0=NORMAL, 1=ABNORMAL) for a class name."""
    return 0 if is_normal_class(class_name) else 1
