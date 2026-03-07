"""
Dataset classes for cervical cancer cell classification.
Supports both SIPaKMeD and Herlev datasets with flexible label mapping.
"""

import os
import torch
from torch.utils.data import Dataset, random_split
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import cv2
from collections import Counter

from utils.label_mapping import LabelMapper, DatasetType, ClassificationMode


class CervicalCellDataset(Dataset):
    """
    Base dataset class for cervical cell classification.
    Supports both SIPaKMeD and Herlev datasets.
    """
    
    def __init__(
        self,
        root_dir: str,
        dataset_type: DatasetType,
        label_mapper: LabelMapper,
        transform: Optional[Callable] = None,
        mode: str = "train"
    ):
        """
        Initialize dataset.
        
        Args:
            root_dir: Root directory containing dataset
            dataset_type: Type of dataset (SIPaKMeD or Herlev)
            label_mapper: Label mapping instance
            transform: Image transformations
            mode: Dataset mode ("train", "val", "test")
        """
        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.label_mapper = label_mapper
        self.transform = transform
        self.mode = mode
        
        self.samples = []
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset from directory structure."""
        # Handle archive folders
        if self.dataset_type == DatasetType.SIPAKMED:
            dataset_dir = os.path.join(self.root_dir, "sipakmed", "archive")
        elif self.dataset_type == DatasetType.HERLEV:
            dataset_dir = os.path.join(self.root_dir, "herlev", "archive (1)")
        else:
            dataset_dir = os.path.join(self.root_dir, self.dataset_type.value)
        
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        
        # Expected structure: dataset/class_name/images/
        class_names = self.label_mapper.get_dataset_classes(self.dataset_type)
        
        for class_id, class_name in class_names.items():
            # Try multiple possible folder names
            possible_folders = [
                str(class_id),
                class_name,
                self._get_archive_class_name(class_id, self.dataset_type)
            ]
            
            class_dir = None
            for folder_name in possible_folders:
                if folder_name:
                    test_dir = os.path.join(dataset_dir, folder_name)
                    if os.path.exists(test_dir):
                        class_dir = test_dir
                        break
            
            if class_dir and os.path.exists(class_dir):
                # Handle nested structure for SIPaKMeD
                if self.dataset_type == DatasetType.SIPAKMED:
                    # Check for nested folder with same name
                    nested_dir = os.path.join(class_dir, os.path.basename(class_dir))
                    if os.path.exists(nested_dir):
                        class_dir = nested_dir
                
                # Load images from the final directory
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                        img_path = os.path.join(class_dir, img_file)
                        # Map label according to current mode
                        mapped_label = self.label_mapper.map_label(class_id, self.dataset_type)
                        self.samples.append((img_path, mapped_label, class_id))
    
    def _get_archive_class_name(self, class_id: int, dataset_type: DatasetType) -> str:
        """Get the actual folder name from archive structure."""
        if dataset_type == DatasetType.SIPAKMED:
            sipakmed_mapping = {
                0: "im_Superficial-Intermediate",  # Superficial-Intermediate
                1: "im_Parabasal",                  # Parabasal
                2: "im_Koilocytotic",               # Koilocytotic
                3: "im_Metaplastic",                # Metaplastic
                4: "im_Dyskeratotic"                # Dyskeratotic
            }
            return sipakmed_mapping.get(class_id)
        
        elif dataset_type == DatasetType.HERLEV:
            herlev_mapping = {
                0: "normal_superficiel",    # Normal superficial (note: typo in archive)
                1: "normal_intermediate",  # Normal intermediate
                2: "normal_columnar",      # Normal columnar
                3: "light_dysplastic",     # Mild dysplasia
                4: "moderate_dysplastic",  # Moderate dysplasia
                5: "severe_dysplastic",    # Severe dysplasia
                6: "carcinoma_in_situ"     # Carcinoma in situ
            }
            return herlev_mapping.get(class_id)
        
        return None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, str]:
        """
        Get sample from dataset.
        
        Returns:
            Tuple of (image, mapped_label, original_label, image_path)
        """
        img_path, mapped_label, original_label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            # Convert PIL to numpy array for albumentations
            image_np = np.array(image)
            # Apply transforms with named argument
            transformed = self.transform(image=image_np)
            image = transformed['image']
        
        return image, mapped_label, original_label, img_path
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get distribution of classes in dataset."""
        distribution = {}
        for _, mapped_label, _ in self.samples:
            distribution[mapped_label] = distribution.get(mapped_label, 0) + 1
        return distribution
    
    def get_sample_paths(self) -> List[str]:
        """Get list of all sample paths."""
        return [sample[0] for sample in self.samples]


class SIPaKMeDDataset(CervicalCellDataset):
    """SIPaKMeD dataset implementation."""
    
    def __init__(
        self,
        root_dir: str,
        label_mapper: LabelMapper,
        transform: Optional[Callable] = None,
        mode: str = "train"
    ):
        super().__init__(
            root_dir=root_dir,
            dataset_type=DatasetType.SIPAKMED,
            label_mapper=label_mapper,
            transform=transform,
            mode=mode
        )


class HerlevDataset(CervicalCellDataset):
    """Herlev dataset implementation."""
    
    def __init__(
        self,
        root_dir: str,
        label_mapper: LabelMapper,
        transform: Optional[Callable] = None,
        mode: str = "train"
    ):
        super().__init__(
            root_dir=root_dir,
            dataset_type=DatasetType.HERLEV,
            label_mapper=label_mapper,
            transform=transform,
            mode=mode
        )


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> torch.utils.data.DataLoader:
    """
    Create DataLoader for dataset.
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        
    Returns:
        DataLoader instance
    """
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )


def get_dataset_info(dataset: CervicalCellDataset) -> Dict:
    """
    Get comprehensive dataset information.
    
    Args:
        dataset: Dataset instance
        
    Returns:
        Dictionary with dataset statistics
    """
    info = {
        'dataset_type': dataset.dataset_type.value,
        'mode': dataset.mode,
        'num_samples': len(dataset),
        'num_classes': dataset.label_mapper.get_num_classes(dataset.dataset_type),
        'class_names': dataset.label_mapper.get_class_names(dataset.dataset_type),
        'class_distribution': dataset.get_class_distribution()
    }
    
    return info


# Factory function to create datasets
def create_dataset(
    root_dir: str,
    dataset_name: str,
    classification_mode: str = "multiclass",
    transform: Optional[Callable] = None,
    mode: str = "train"
) -> CervicalCellDataset:
    """
    Factory function to create dataset instances.
    
    Args:
        root_dir: Root directory containing datasets
        dataset_name: Name of dataset ("sipakmed" or "herlev")
        classification_mode: "binary" or "multiclass"
        transform: Image transformations
        mode: Dataset mode
        
    Returns:
        Dataset instance
    """
    # Create label mapper
    if classification_mode == "binary":
        label_mapper = LabelMapper(ClassificationMode.BINARY)
    else:
        label_mapper = LabelMapper(ClassificationMode.MULTICLASS)
    
    # Create dataset based on name
    if dataset_name.lower() == "sipakmed":
        return SIPaKMeDDataset(root_dir, label_mapper, transform, mode)
    elif dataset_name.lower() == "herlev":
        return HerlevDataset(root_dir, label_mapper, transform, mode)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


class DatasetManager:
    """
    Comprehensive dataset manager for cervical cancer classification.
    Handles loading, splitting, and creating data loaders for both datasets.
    """
    
    def __init__(
        self,
        root_dir: str,
        classification_mode: str = "multiclass",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42
    ):
        """
        Initialize dataset manager.
        
        Args:
            root_dir: Root directory containing datasets
            classification_mode: "binary" or "multiclass"
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing
            random_seed: Random seed for reproducibility
        """
        self.root_dir = root_dir
        self.classification_mode = classification_mode
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")
        
        # Create label mapper
        if classification_mode == "binary":
            self.label_mapper = LabelMapper(ClassificationMode.BINARY)
        else:
            self.label_mapper = LabelMapper(ClassificationMode.MULTICLASS)
        
        # Initialize datasets
        self.sipakmed_dataset = None
        self.herlev_dataset = None
        self.train_datasets = {}
        self.val_datasets = {}
        self.test_datasets = {}
    
    def load_datasets(self):
        """Load both SIPaKMeD and Herlev datasets."""
        print("Loading datasets...")
        
        try:
            # Load SIPaKMeD
            self.sipakmed_dataset = SIPaKMeDDataset(
                root_dir=self.root_dir,
                label_mapper=self.label_mapper,
                transform=None,
                mode="all"
            )
            print(f"✅ SIPaKMeD loaded: {len(self.sipakmed_dataset)} samples")
        except FileNotFoundError as e:
            print(f"❌ SIPaKMeD not found: {e}")
            self.sipakmed_dataset = None
        
        try:
            # Load Herlev
            self.herlev_dataset = HerlevDataset(
                root_dir=self.root_dir,
                label_mapper=self.label_mapper,
                transform=None,
                mode="all"
            )
            print(f"✅ Herlev loaded: {len(self.herlev_dataset)} samples")
        except FileNotFoundError as e:
            print(f"❌ Herlev not found: {e}")
            self.herlev_dataset = None
        
        if not self.sipakmed_dataset and not self.herlev_dataset:
            raise ValueError("No datasets found!")
    
    def create_splits(self):
        """Create train/val/test splits for loaded datasets."""
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        for dataset_name, dataset in [("sipakmed", self.sipakmed_dataset), ("herlev", self.herlev_dataset)]:
            if dataset is None:
                continue
            
            print(f"Creating splits for {dataset_name}...")
            
            # Calculate split sizes
            total_size = len(dataset)
            train_size = int(total_size * self.train_ratio)
            val_size = int(total_size * self.val_ratio)
            test_size = total_size - train_size - val_size
            
            # Create splits
            train_dataset, val_dataset, test_dataset = random_split(
                dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(self.random_seed)
            )
            
            # Store splits
            self.train_datasets[dataset_name] = train_dataset
            self.val_datasets[dataset_name] = val_dataset
            self.test_datasets[dataset_name] = test_dataset
            
            print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    def create_data_loaders(
        self,
        train_transform,
        val_transform,
        batch_size: int = 32,
        num_workers: int = 4
    ) -> Dict[str, torch.utils.data.DataLoader]:
        """
        Create data loaders for all splits.
        
        Args:
            train_transform: Transformations for training data
            val_transform: Transformations for validation/test data
            batch_size: Batch size
            num_workers: Number of worker processes
            
        Returns:
            Dictionary of data loaders
        """
        data_loaders = {}
        
        for dataset_name in ["sipakmed", "herlev"]:
            if dataset_name not in self.train_datasets:
                continue
            
            print(f"Creating data loaders for {dataset_name}...")
            
            # Create datasets with transforms
            train_dataset = TransformDataset(
                self.train_datasets[dataset_name],
                train_transform
            )
            val_dataset = TransformDataset(
                self.val_datasets[dataset_name],
                val_transform
            )
            test_dataset = TransformDataset(
                self.test_datasets[dataset_name],
                val_transform
            )
            
            # Create data loaders
            data_loaders[f"{dataset_name}_train"] = create_dataloader(
                train_dataset, batch_size, shuffle=True, num_workers=num_workers
            )
            data_loaders[f"{dataset_name}_val"] = create_dataloader(
                val_dataset, batch_size, shuffle=False, num_workers=num_workers
            )
            data_loaders[f"{dataset_name}_test"] = create_dataloader(
                test_dataset, batch_size, shuffle=False, num_workers=num_workers
            )
        
        return data_loaders
    
    def get_dataset_summary(self) -> Dict:
        """Get comprehensive dataset summary."""
        if self.classification_mode == ClassificationMode.BINARY:
            num_classes = self.label_mapper.get_num_classes()
            class_names = self.label_mapper.get_class_names()
        else:  # MULTICLASS
            # For multiclass, we need to handle different datasets separately
            # Use SIPaKMeD as default for overall summary
            num_classes = self.label_mapper.get_num_classes(DatasetType.SIPAKMED)
            class_names = self.label_mapper.get_class_names(DatasetType.SIPAKMED)
        
        summary = {
            "classification_mode": self.classification_mode,
            "num_classes": num_classes,
            "class_names": class_names,
            "datasets": {}
        }
        
        for dataset_name, dataset in [("sipakmed", self.sipakmed_dataset), ("herlev", self.herlev_dataset)]:
            if dataset is None:
                continue
            
            # Get class distribution
            distribution = dataset.get_class_distribution()
            
            # Get split sizes
            train_size = len(self.train_datasets[dataset_name]) if dataset_name in self.train_datasets else 0
            val_size = len(self.val_datasets[dataset_name]) if dataset_name in self.val_datasets else 0
            test_size = len(self.test_datasets[dataset_name]) if dataset_name in self.test_datasets else 0
            
            summary["datasets"][dataset_name] = {
                "total_samples": len(dataset),
                "train_samples": train_size,
                "val_samples": val_size,
                "test_samples": test_size,
                "class_distribution": distribution,
                "most_common_class": max(distribution.items(), key=lambda x: x[1]) if distribution else None,
                "least_common_class": min(distribution.items(), key=lambda x: x[1]) if distribution else None
            }
        
        return summary
    
    def print_summary(self):
        """Print detailed dataset summary."""
        summary = self.get_dataset_summary()
        
        print("\n" + "="*80)
        print("DATASET SUMMARY")
        print("="*80)
        print(f"Classification Mode: {summary['classification_mode'].upper()}")
        print(f"Number of Classes: {summary['num_classes']}")
        print(f"Class Names: {summary['class_names']}")
        print()
        
        for dataset_name, info in summary["datasets"].items():
            print(f"{dataset_name.upper()} DATASET:")
            print(f"  Total Samples: {info['total_samples']}")
            print(f"  Train/Val/Test Split: {info['train_samples']}/{info['val_samples']}/{info['test_samples']}")
            print(f"  Class Distribution:")
            
            for class_id, count in info["class_distribution"].items():
                class_name = summary["class_names"][class_id] if class_id < len(summary["class_names"]) else f"Class_{class_id}"
                percentage = (count / info["total_samples"]) * 100
                print(f"    {class_name}: {count} ({percentage:.1f}%)")
            
            if info["most_common_class"]:
                most_class_id, most_count = info["most_common_class"]
                least_class_id, least_count = info["least_common_class"]
                most_class_name = summary["class_names"][most_class_id] if most_class_id < len(summary["class_names"]) else f"Class_{most_class_id}"
                least_class_name = summary["class_names"][least_class_id] if least_class_id < len(summary["class_names"]) else f"Class_{least_class_id}"
                print(f"  Most Common: {most_class_name} ({most_count})")
                print(f"  Least Common: {least_class_name} ({least_count})")
            print()
        
        print("="*80)


class TransformDataset(Dataset):
    """
    Wrapper dataset that applies transforms to a subset dataset.
    """
    
    def __init__(self, subset_dataset, transform):
        """
        Initialize transform dataset.
        
        Args:
            subset_dataset: Subset dataset from random_split
            transform: Transform to apply
        """
        self.subset_dataset = subset_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.subset_dataset)
    
    def __getitem__(self, idx):
        # Get original sample
        img_path, mapped_label, original_label = self.subset_dataset.dataset.samples[self.subset_dataset.indices[idx]]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transform
        if self.transform:
            if isinstance(self.transform, dict):
                # Handle albumentations transforms
                image = self.transform(image=np.array(image))['image']
            else:
                image = self.transform(image)
        
        return image, mapped_label, original_label, img_path


if __name__ == "__main__":
    # Example usage
    from preprocessing.transforms import get_train_transforms, get_val_transforms
    
    # Create label mapper
    label_mapper = LabelMapper(ClassificationMode.BINARY)
    
    # Create datasets
    train_dataset = SIPaKMeDDataset(
        root_dir="datasets",
        label_mapper=label_mapper,
        transform=get_train_transforms(),
        mode="train"
    )
    
    print(f"Dataset info: {get_dataset_info(train_dataset)}")
    print(f"Number of samples: {len(train_dataset)}")
