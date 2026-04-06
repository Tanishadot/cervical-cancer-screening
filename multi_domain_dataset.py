"""
Multi-domain dataset management for robust cross-dataset training.
Handles SIPaKMeD + Herlev training with proper splitting and class balancing.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from PIL import Image
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Callable
from collections import Counter

from utils.label_mapping import LabelMapper, ClassificationMode, DatasetType
from datasets.dataset import HerlevDataset, SIPaKMeDDataset
from preprocessing.transforms import get_train_transforms, get_val_transforms


class MultiDomainDataset(Dataset):
    """
    Combined dataset for multi-domain training (SIPaKMeD + Herlev).
    """
    
    def __init__(
        self,
        sipakmed_dataset: Optional[Dataset] = None,
        herlev_dataset: Optional[Dataset] = None,
        transform: Optional[Callable] = None
    ):
        """
        Initialize multi-domain dataset.
        
        Args:
            sipakmed_dataset: SIPaKMeD dataset
            herlev_dataset: Herlev dataset (train or test)
            transform: Transform to apply
        """
        self.sipakmed_dataset = sipakmed_dataset
        self.herlev_dataset = herlev_dataset
        self.transform = transform
        
        # Combine samples from both datasets
        self.samples = []
        self.dataset_sources = []
        
        if sipakmed_dataset is not None:
            # Use transformed dataset directly if available
            if hasattr(sipakmed_dataset, 'dataset') and hasattr(sipakmed_dataset, 'indices'):
                # This is a TransformedDataset, use it directly
                for i in range(len(sipakmed_dataset)):
                    self.samples.append(sipakmed_dataset[i])  # This will apply transforms
                    self.dataset_sources.append('sipakmed')
            else:
                # Direct dataset access
                for i, (img_path, label, orig_label) in enumerate(sipakmed_dataset.samples):
                    self.samples.append((img_path, label, orig_label))
                    self.dataset_sources.append('sipakmed')
        
        if herlev_dataset is not None:
            # Use transformed dataset directly if available
            if hasattr(herlev_dataset, 'dataset') and hasattr(herlev_dataset, 'indices'):
                # This is a TransformedDataset, use it directly
                for i in range(len(herlev_dataset)):
                    self.samples.append(herlev_dataset[i])  # This will apply transforms
                    self.dataset_sources.append('herlev')
            else:
                # Direct dataset access
                for i, (img_path, label, orig_label) in enumerate(herlev_dataset.samples):
                    self.samples.append((img_path, label, orig_label))
                    self.dataset_sources.append('herlev')
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Multi-domain dataset created: {len(self.samples)} samples")
        self.logger.info(f"SIPaKMeD: {len(sipakmed_dataset) if sipakmed_dataset else 0} samples")
        self.logger.info(f"Herlev: {len(herlev_dataset) if herlev_dataset else 0} samples")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple:
        if idx == 0:
            print(f"DEBUG [MultiDomainDataset-START]: idx={idx}, transform={self.transform is not None}")
        
        sample = self.samples[idx]
        
        # Always apply transforms if available
        if len(sample) == 4 and isinstance(sample[0], torch.Tensor):
            # Already transformed (tensor, label, orig_label, path)
            image, label, orig_label, img_path = sample
            return image, label, orig_label, img_path, self.dataset_sources[idx]
        else:
            # Raw sample (img_path, label, orig_label), need to apply transforms
            img_path, label, orig_label = sample
            
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                if hasattr(self.transform, '__call__') and hasattr(self.transform, 'transforms'):
                    # Albumentations transform - pass as named argument
                    image_np = np.array(image)
                    transformed = self.transform(image=image_np)
                    image = transformed['image']  # This should be a tensor
                else:
                    # Regular torchvision transform
                    image = self.transform(image)
            
            # Safety checks (MANDATORY)
            assert isinstance(image, torch.Tensor), f"Image is not tensor, got {type(image)}"
            assert image.shape[0] == 3, f"Channel mismatch, expected 3, got {image.shape[0]}"
            assert image.shape[1] == 224, f"Height mismatch, expected 224, got {image.shape[1]}"
            assert image.shape[2] == 224, f"Width mismatch, expected 224, got {image.shape[2]}"
            assert image.dtype == torch.float32, f"Dtype mismatch, expected float32, got {image.dtype}"
            
            # Debug print for first batch only
            if idx == 0:
                print(f"DEBUG [MultiDomainDataset]: Applied transforms, type={type(image)}, shape={image.shape}, dtype={image.dtype}")
            
            return image, label, orig_label, img_path, self.dataset_sources[idx]
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get class distribution."""
        distribution = Counter()
        for _, label, _ in self.samples:
            distribution[label] += 1
        return dict(distribution)
    
    def get_dataset_distribution(self) -> Dict[str, int]:
        """Get dataset source distribution."""
        distribution = Counter(self.dataset_sources)
        return dict(distribution)


class MultiDomainManager:
    """
    Manager for multi-domain training and evaluation.
    """
    
    def __init__(
        self,
        root_dir: str,
        classification_mode: str = "binary",
        herlev_train_ratio: float = 0.2,
        random_seed: int = 42,
        use_cropped_only: bool = False
    ):
        """
        Initialize multi-domain manager.
        
        Args:
            root_dir: Root directory containing datasets
            classification_mode: "binary" or "multiclass"
            herlev_train_ratio: Ratio of Herlev data for training
            random_seed: Random seed for reproducibility
            use_cropped_only: Whether to use only cropped images
        """
        self.root_dir = root_dir
        self.classification_mode = classification_mode
        self.herlev_train_ratio = herlev_train_ratio
        self.random_seed = random_seed
        self.use_cropped_only = use_cropped_only
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Create label mapper
        if classification_mode == "binary":
            self.label_mapper = LabelMapper(ClassificationMode.BINARY)
        else:
            self.label_mapper = LabelMapper(ClassificationMode.MULTICLASS)
        
        # Initialize datasets
        self.sipakmed_dataset = None
        self.herlev_train_dataset = None
        self.herlev_test_dataset = None
        
        # Set random seeds
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    def load_datasets(self):
        """Load and split datasets for multi-domain training."""
        self.logger.info("Loading datasets for multi-domain training...")
        
        # 1. Load SIPaKMeD (full dataset for training)
        try:
            self.sipakmed_dataset = SIPaKMeDDataset(
                root_dir=self.root_dir,
                label_mapper=self.label_mapper,
                transform=None,
                mode="train",
                use_cropped_only=self.use_cropped_only
            )
            self.logger.info(f"✅ SIPaKMeD loaded: {len(self.sipakmed_dataset)} samples")
        except FileNotFoundError as e:
            self.logger.error(f"❌ SIPaKMeD not found: {e}")
            self.sipakmed_dataset = None
        
        # 2. Load and split Herlev dataset
        try:
            # Load full Herlev dataset
            herlev_full = HerlevDataset(
                root_dir=self.root_dir,
                label_mapper=self.label_mapper,
                transform=None,
                mode="all",
                use_cropped_only=False  # Use all images for Herlev
            )
            
            self.logger.info(f"✅ Herlev full dataset loaded: {len(herlev_full)} samples")
            
            # Split Herlev into train and test
            herlev_indices = list(range(len(herlev_full)))
            herlev_train_indices, herlev_test_indices = train_test_split(
                herlev_indices,
                test_size=1 - self.herlev_train_ratio,
                random_state=self.random_seed,
                stratify=[herlev_full.samples[i][1] for i in herlev_indices]  # Stratify by label
            )
            
            # Create Herlev train and test datasets
            self.herlev_train_dataset = self._create_subset(herlev_full, herlev_train_indices, "herlev_train")
            self.herlev_test_dataset = self._create_subset(herlev_full, herlev_test_indices, "herlev_test")
            
            self.logger.info(f"✅ Herlev split - Train: {len(self.herlev_train_dataset)}, Test: {len(self.herlev_test_dataset)}")
            
        except FileNotFoundError as e:
            self.logger.error(f"❌ Herlev not found: {e}")
            self.herlev_train_dataset = None
            self.herlev_test_dataset = None
        
        if not self.sipakmed_dataset and not self.herlev_train_dataset:
            self.logger.error("No training datasets found!")
            raise ValueError("No training datasets found!")
    
    def _create_subset(self, full_dataset: Dataset, indices: List[int], name: str) -> Dataset:
        """Create a subset dataset from indices."""
        subset = torch.utils.data.Subset(full_dataset, indices)
        
        # Create a new dataset with proper metadata
        class SubsetDataset(Dataset):
            def __init__(self, subset, name):
                self.subset = subset
                self.name = name
                self.samples = [full_dataset.samples[i] for i in indices]
            
            def __len__(self):
                return len(self.subset)
            
            def __getitem__(self, idx):
                return self.subset[idx]
            
            def get_class_distribution(self):
                """Get class distribution for subset."""
                distribution = {}
                for _, label, _ in self.samples:
                    distribution[label] = distribution.get(label, 0) + 1
                return distribution
        
        return SubsetDataset(subset, name)
    
    def create_training_datasets(self, train_transform) -> MultiDomainDataset:
        """
        Create combined training dataset.
        
        Args:
            train_transform: Training transforms
            
        Returns:
            Combined multi-domain training dataset
        """
        self.logger.info("Creating multi-domain training dataset...")
        
        # Create datasets with transforms
        sipakmed_transformed = None
        if self.sipakmed_dataset:
            sipakmed_transformed = self._create_transformed_dataset(
                self.sipakmed_dataset, train_transform, "sipakmed"
            )
        
        herlev_transformed = None
        if self.herlev_train_dataset:
            herlev_transformed = self._create_transformed_dataset(
                self.herlev_train_dataset, train_transform, "herlev_train"
            )
        
        # Create combined dataset (no transform since already applied)
        combined_dataset = MultiDomainDataset(
            sipakmed_dataset=sipakmed_transformed,
            herlev_dataset=herlev_transformed,
            transform=None  # Transforms already applied
        )
        
        self.logger.info(f"✅ Combined training dataset: {len(combined_dataset)} samples")
        self.logger.info(f"Class distribution: {combined_dataset.get_class_distribution()}")
        self.logger.info(f"Dataset distribution: {combined_dataset.get_dataset_distribution()}")
        
        return combined_dataset
    
    def create_test_dataset(self, val_transform) -> HerlevDataset:
        """
        Create test dataset (Herlev only).
        
        Args:
            val_transform: Validation transforms
            
        Returns:
            Herlev test dataset
        """
        if not self.herlev_test_dataset:
            raise ValueError("Herlev test dataset not available!")
        
        self.logger.info("Creating Herlev test dataset...")
        
        test_dataset = self._create_transformed_dataset(
            self.herlev_test_dataset, val_transform, "herlev_test"
        )
        
        self.logger.info(f"✅ Test dataset: {len(test_dataset)} samples")
        
        return test_dataset
    
    def _create_transformed_dataset(self, dataset: Dataset, transform, name: str) -> Dataset:
        """Create dataset with transforms applied."""
        class TransformedDataset(Dataset):
            def __init__(self, original_dataset, transform, name):
                self.original_dataset = original_dataset
                self.transform = transform
                self.name = name
                self.samples = original_dataset.samples
            
            def __len__(self):
                return len(self.original_dataset)
            
            def __getitem__(self, idx):
                img_path, label, orig_label = self.original_dataset.samples[idx]
                
                # Load image
                image = Image.open(img_path).convert('RGB')
                
                # Apply transforms
                if self.transform:
                    if hasattr(self.transform, '__call__') and hasattr(self.transform, 'transforms'):
                        # Albumentations transform - pass as named argument
                        image_np = np.array(image)
                        transformed = self.transform(image=image_np)
                        image = transformed['image']  # This should be a tensor
                    else:
                        # Regular torchvision transform
                        image = self.transform(image)
                
                # Safety checks (MANDATORY)
                assert isinstance(image, torch.Tensor), f"Image is not tensor, got {type(image)}"
                assert image.shape[0] == 3, f"Channel mismatch, expected 3, got {image.shape[0]}"
                assert image.shape[1] == 224, f"Height mismatch, expected 224, got {image.shape[1]}"
                assert image.shape[2] == 224, f"Width mismatch, expected 224, got {image.shape[2]}"
                assert image.dtype == torch.float32, f"Dtype mismatch, expected float32, got {image.dtype}"
                
                # Debug print for first batch only
                if idx == 0:
                    print(f"DEBUG [TransformedDataset-{self.name}]: type={type(image)}, shape={image.shape}, dtype={image.dtype}")
                
                return image, label, orig_label, img_path
        
        return TransformedDataset(dataset, transform, name)
    
    def create_balanced_sampler(self, dataset: MultiDomainDataset) -> WeightedRandomSampler:
        """
        Create balanced sampler for class-imbalanced dataset.
        
        Args:
            dataset: Training dataset
            
        Returns:
            WeightedRandomSampler for balanced sampling
        """
        # Calculate class weights
        class_counts = dataset.get_class_distribution()
        total_samples = len(dataset)
        
        # Calculate weights inversely proportional to class frequency
        class_weights = {}
        for class_id, count in class_counts.items():
            weight = total_samples / (len(class_counts) * count)
            class_weights[class_id] = weight
        
        # Assign weights to each sample
        sample_weights = []
        for _, label, _ in dataset.samples:
            sample_weights.append(class_weights[label])
        
        # Create sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        self.logger.info(f"Class weights: {class_weights}")
        
        return sampler
    
    def create_data_loaders(
        self,
        train_transform,
        val_transform,
        batch_size: int = 32,
        num_workers: int = 4
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create training and test data loaders.
        
        Args:
            train_transform: Training transforms
            val_transform: Validation transforms
            batch_size: Batch size
            num_workers: Number of workers
            
        Returns:
            Tuple of (train_loader, test_loader)
        """
        # Create training dataset
        train_dataset = self.create_training_datasets(train_transform)
        
        # Create balanced sampler
        train_sampler = self.create_balanced_sampler(train_dataset)
        
        # Create training data loader
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        # Create test dataset and loader
        test_dataset = self.create_test_dataset(val_transform)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        self.logger.info(f"✅ Data loaders created - Train: {len(train_loader)} batches, Test: {len(test_loader)} batches")
        
        return train_loader, test_loader
    
    def get_dataset_summary(self) -> Dict:
        """Get comprehensive dataset summary."""
        summary = {
            "classification_mode": self.classification_mode,
            "herlev_train_ratio": self.herlev_train_ratio,
            "datasets": {}
        }
        
        # SIPaKMeD info
        if self.sipakmed_dataset:
            summary["datasets"]["sipakmed"] = {
                "total_samples": len(self.sipakmed_dataset),
                "class_distribution": self.sipakmed_dataset.get_class_distribution()
            }
        
        # Herlev info
        if self.herlev_train_dataset and self.herlev_test_dataset:
            summary["datasets"]["herlev"] = {
                "total_samples": len(self.herlev_train_dataset) + len(self.herlev_test_dataset),
                "train_samples": len(self.herlev_train_dataset),
                "test_samples": len(self.herlev_test_dataset),
                "train_distribution": self.herlev_train_dataset.get_class_distribution(),
                "test_distribution": self.herlev_test_dataset.get_class_distribution()
            }
        
        return summary
    
    def print_summary(self):
        """Print detailed dataset summary."""
        summary = self.get_dataset_summary()
        
        print("\n" + "="*80)
        print("MULTI-DOMAIN DATASET SUMMARY")
        print("="*80)
        print(f"Classification Mode: {summary['classification_mode'].upper()}")
        print(f"Herlev Train Ratio: {summary['herlev_train_ratio']:.1%}")
        print()
        
        for dataset_name, info in summary["datasets"].items():
            print(f"{dataset_name.upper()} DATASET:")
            if "total_samples" in info:
                print(f"  Total Samples: {info['total_samples']}")
            
            if "train_samples" in info:
                print(f"  Train/Test Split: {info['train_samples']}/{info['test_samples']}")
            
            if "class_distribution" in info:
                print(f"  Class Distribution: {info['class_distribution']}")
            
            if "train_distribution" in info:
                print(f"  Train Distribution: {info['train_distribution']}")
                print(f"  Test Distribution: {info['test_distribution']}")
            
            print()
        
        print("="*80)


if __name__ == "__main__":
    # Test multi-domain dataset
    from preprocessing.transforms import get_train_transforms, get_val_transforms
    
    # Create manager
    manager = MultiDomainManager(
        root_dir="datasets",
        classification_mode="binary",
        herlev_train_ratio=0.2,
        random_seed=42
    )
    
    # Load datasets
    manager.load_datasets()
    manager.print_summary()
    
    # Create data loaders
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    
    train_loader, test_loader = manager.create_data_loaders(
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=16
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
