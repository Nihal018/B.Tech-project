"""
Siamese U-Net with Alpha Blending for Disaster Damage Assessment
Optimized for 5-class segmentation with separate test dataset support
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# For reproducibility


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything()

# Configuration


class Config:
    # Data paths - adjust these to your dataset location
    # Contains 'images' and 'labelled_images' folders
    DATA_DIR = '/home/sujit1/dataset_disaster/Dataset/train/'
    TEST_DATA_DIR = '/home/sujit1/dataset_disaster/Dataset/test/'
    # Model parameters
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Training parameters
    TRAIN_BATCH_SIZE = 8  # Will be auto-adjusted for small datasets
    VALID_BATCH_SIZE = 4  # Smaller validation batch size
    TEST_BATCH_SIZE = 4   # Test batch size
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-5
    EPOCHS = 30
    EARLY_STOPPING_PATIENCE = 5
    GRADIENT_ACCUMULATION_STEPS = 1

    # Progressive resizing strategy
    PROGRESSIVE_RESIZING = True
    INITIAL_IMG_SIZE = 256
    FINAL_IMG_SIZE = 512
    PROGRESSIVE_EPOCHS = 15

    # Final image size
    IMG_SIZE = 512

    # Mixed precision
    USE_AMP = True

    # Class weights for 5 classes (black, white, green, orange, red)
    CLASS_WEIGHTS = [0.1, 0.5, 1.0, 2.0, 2.5]

    # Number of damage classes (5 classes)
    NUM_CLASSES = 5

    # Alpha blending parameters
    ALPHA_INIT = 0.5
    LEARNABLE_ALPHA = True

    # Model complexity
    BASE_FILTERS = 32
    DEPTH = 3  # Reduced depth for faster training and smaller model

    # Checkpoint saving frequency
    CHECKPOINT_FREQ = 1

    # Verbose output
    VERBOSE = True  # Print detailed information

    # Small dataset handling
    MIN_STEPS_PER_EPOCH = 10  # Minimum steps per epoch for scheduler
    SMALL_DATASET_THRESHOLD = 20  # Threshold for considering a dataset as small

    # Test results
    SAVE_TEST_RESULTS = True
    TEST_RESULTS_DIR = 'test_results'

# Checkpoint manager


class CheckpointManager:
    def __init__(self, save_dir='checkpoints', model_name='siamese_unet'):
        self.save_dir = Path(save_dir)
        self.model_name = model_name

        # Create checkpoint directory if it doesn't exist
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Path to metadata file containing training state
        self.metadata_path = self.save_dir / f"{model_name}_metadata.json"

    def save_checkpoint(self, model, optimizer, scheduler, epoch, stage_idx,
                        curr_stage_epoch, best_valid_loss, early_stopping_counter,
                        config, stage=None):
        """
        Save a complete checkpoint with model weights and training state
        """
        # Create checkpoint filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self.save_dir / \
            f"{self.model_name}_epoch{epoch}_{timestamp}.pth"

        # Save model state, optimizer state, and training parameters
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'epoch': epoch,
            'stage_idx': stage_idx,
            'curr_stage_epoch': curr_stage_epoch,
            'best_valid_loss': best_valid_loss,
            'early_stopping_counter': early_stopping_counter,
            'config': {k: v for k, v in vars(config).items() if not k.startswith('__')}
        }

        if stage:
            checkpoint['stage'] = stage

        # Save the checkpoint
        torch.save(checkpoint, checkpoint_path)

        # Update metadata with information about the latest checkpoint
        metadata = {
            'latest_checkpoint': str(checkpoint_path),
            'epoch': epoch,
            'stage_idx': stage_idx,
            'curr_stage_epoch': curr_stage_epoch,
            'best_valid_loss': best_valid_loss,
            'early_stopping_counter': early_stopping_counter,
            'timestamp': timestamp,
            'config': {k: v for k, v in vars(config).items() if not k.startswith('__')}
        }

        # Save metadata
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f"Checkpoint saved: {checkpoint_path}")

        return checkpoint_path

    def load_latest_checkpoint(self, model=None, optimizer=None, scheduler=None):
        """
        Load the latest checkpoint and return training state
        """
        # Check if metadata file exists
        if not self.metadata_path.exists():
            print("No checkpoint metadata found. Starting fresh training.")
            return None

        # Load metadata to find the latest checkpoint
        try:
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)

            latest_checkpoint_path = metadata['latest_checkpoint']

            # Check if the checkpoint file actually exists
            if not os.path.exists(latest_checkpoint_path):
                print(
                    f"Checkpoint file {latest_checkpoint_path} not found. Starting fresh training.")
                return None

            # Load the checkpoint
            checkpoint = torch.load(latest_checkpoint_path, map_location='cpu')

            # Load model state if model is provided
            if model:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Model weights loaded from {latest_checkpoint_path}")

            # Load optimizer state if optimizer is provided
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # Move optimizer state to the appropriate device if using GPU
                if torch.cuda.is_available():
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda()
                print("Optimizer state loaded")

            # Load scheduler state if scheduler is provided
            if scheduler and checkpoint.get('scheduler_state_dict'):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("Scheduler state loaded")

            print(
                f"Resuming from epoch {checkpoint['epoch']}, stage {checkpoint['stage_idx']}")

            return checkpoint

        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting fresh training.")
            return None

    def find_best_model(self):
        """
        Find the best model based on validation loss
        """
        # Check if metadata file exists
        if not self.metadata_path.exists():
            print("No checkpoint metadata found.")
            return None

        # Load metadata to find the best checkpoint
        try:
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)

            # The metadata contains the path to the best model
            best_model_path = self.save_dir / f"{self.model_name}_best.pth"

            if os.path.exists(best_model_path):
                return best_model_path
            else:
                return metadata['latest_checkpoint']

        except Exception as e:
            print(f"Error finding best model: {e}")
            return None

# Base model components


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Ensure correct size for concatenation
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX //
                   2, diffY // 2, diffY - diffY // 2])

        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class AlphaBlending(nn.Module):
    def __init__(self, features_size, alpha_init=0.5, learnable=True):
        super().__init__()
        self.learnable = learnable

        if learnable:
            # Create a learnable parameter for each feature map
            self.alpha = nn.Parameter(torch.ones(features_size) * alpha_init)
            # Ensure alpha is between 0 and 1 using sigmoid
            self.sigmoid = nn.Sigmoid()
        else:
            self.alpha = alpha_init

    def forward(self, pre_features, post_features):
        if self.learnable:
            # Apply sigmoid to ensure alpha is between 0 and 1
            alpha = self.sigmoid(self.alpha)

            # Reshape alpha to match feature dimensions for element-wise multiplication
            alpha_expanded = alpha.view(*alpha.shape, 1, 1)

            # Blend features with learned alpha
            blended_features = alpha_expanded * post_features + \
                (1 - alpha_expanded) * pre_features
        else:
            # Static alpha blending
            blended_features = self.alpha * post_features + \
                (1 - self.alpha) * pre_features

        return blended_features


class SiameseEncoder(nn.Module):
    def __init__(self, n_channels=3, base_filters=32, depth=4):
        super().__init__()
        self.n_channels = n_channels
        self.depth = depth

        # Define filter sizes based on depth
        filters = [base_filters]
        for i in range(depth):
            filters.append(filters[-1] * 2)

        # Input conv
        self.inc = DoubleConv(n_channels, filters[0])

        # Downsampling path
        self.down_layers = nn.ModuleList([
            Down(filters[i], filters[i+1]) for i in range(depth)
        ])

    def forward(self, x):
        features = []

        # Input conv
        x = self.inc(x)
        features.append(x)

        # Downsampling path
        for down in self.down_layers:
            x = down(x)
            features.append(x)

        return features


class SiameseUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_channels = 3
        self.n_classes = config.NUM_CLASSES
        self.bilinear = True
        self.depth = config.DEPTH
        self.base_filters = config.BASE_FILTERS

        # Calculate filter sizes
        filters = [self.base_filters]
        for i in range(self.depth):
            filters.append(filters[-1] * 2)

        # Shared encoder for both pre and post disaster images
        self.encoder = SiameseEncoder(
            n_channels=self.n_channels,
            base_filters=self.base_filters,
            depth=self.depth
        )

        # Alpha blending modules for each level of features
        self.alpha_blends = nn.ModuleList([
            AlphaBlending([filters[i]], config.ALPHA_INIT,
                          config.LEARNABLE_ALPHA)
            for i in range(self.depth + 1)
        ])

        # Decoder path with skip connections
        self.up_layers = nn.ModuleList()

        # Create up-sampling blocks dynamically based on depth
        for i in range(self.depth):
            in_channels = filters[self.depth-i] + filters[self.depth-i-1]
            out_channels = filters[self.depth-i-1]
            self.up_layers.append(
                Up(in_channels, out_channels, bilinear=self.bilinear))

        # Output convolution
        self.outc = OutConv(filters[0], self.n_classes)

        # Apply weight initialization for faster convergence
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, pre_img, post_img):
        # Encode both pre and post disaster images
        pre_features = self.encoder(pre_img)
        post_features = self.encoder(post_img)

        # Alpha blending at each feature level
        blended_features = []
        for i in range(len(pre_features)):
            blended_features.append(
                self.alpha_blends[i](pre_features[i], post_features[i])
            )

        # Decoder with skip connections
        x = blended_features[-1]

        for i, up in enumerate(self.up_layers):
            skip_index = self.depth - i - 1
            x = up(x, blended_features[skip_index])

        # Final output layer
        logits = self.outc(x)

        return logits

# Dataset class


class DisasterDataset(Dataset):
    def __init__(self, pre_imgs, post_imgs, masks=None, transform=None):
        self.pre_imgs = pre_imgs
        self.post_imgs = post_imgs
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.pre_imgs)

    def __getitem__(self, idx):
        pre_img_path = self.pre_imgs[idx]
        post_img_path = self.post_imgs[idx]

        pre_img = np.array(Image.open(pre_img_path).convert('RGB'))
        post_img = np.array(Image.open(post_img_path).convert('RGB'))

        # For training with masks
        if self.masks is not None:
            mask_path = self.masks[idx]
            mask = np.array(Image.open(mask_path))

            # Convert RGB mask to class indices if needed
            if len(mask.shape) == 3 and mask.shape[2] == 3:
                # Convert to class indices for 5 classes
                temp_mask = np.zeros(mask.shape[:2], dtype=np.uint8)

                # Class 0: Background (black)
                temp_mask[(mask[:, :, 0] < 10) & (
                    mask[:, :, 1] < 10) & (mask[:, :, 2] < 10)] = 0

                # Class 1: Undamaged (white)
                temp_mask[(mask[:, :, 0] > 200) & (
                    mask[:, :, 1] > 200) & (mask[:, :, 2] > 200)] = 1

                # Class 2: Minor damage (green)
                temp_mask[(mask[:, :, 0] < 100) & (
                    mask[:, :, 1] > 200) & (mask[:, :, 2] < 100)] = 2

                # Class 3: Major damage (orange)
                temp_mask[(mask[:, :, 0] > 200) & (mask[:, :, 1] > 100) & (
                    mask[:, :, 1] < 200) & (mask[:, :, 2] < 100)] = 3

                # Class 4: Destroyed (red)
                temp_mask[(mask[:, :, 0] > 200) & (
                    mask[:, :, 1] < 100) & (mask[:, :, 2] < 100)] = 4

                mask = temp_mask

            # Apply transformations
            if self.transform:
                augmented = self.transform(
                    image=pre_img, image2=post_img, mask=mask)
                pre_img = augmented['image']
                post_img = augmented['image2']
                mask = augmented['mask']

            # Convert to tensors if not done by transform
            if not isinstance(pre_img, torch.Tensor):
                pre_img = torch.from_numpy(pre_img.transpose(2, 0, 1)).float()
                post_img = torch.from_numpy(
                    post_img.transpose(2, 0, 1)).float()
                # Ensure Long type for masks
                mask = torch.from_numpy(mask).long()
            else:
                # Ensure mask is Long type even if already a tensor
                mask = mask.long()

            return {
                'pre_image': pre_img,
                'post_image': post_img,
                'mask': mask
            }
        else:
            # For inference without masks
            if self.transform:
                augmented = self.transform(image=pre_img, image2=post_img)
                pre_img = augmented['image']
                post_img = augmented['image2']

            if not isinstance(pre_img, torch.Tensor):
                pre_img = torch.from_numpy(pre_img.transpose(2, 0, 1)).float()
                post_img = torch.from_numpy(
                    post_img.transpose(2, 0, 1)).float()

            return {
                'pre_image': pre_img,
                'post_image': post_img
            }

# Data preparation function for training data


def prepare_data(config):
    """
    Prepare data from a directory structure where:
    - Pre and post disaster images are in the "images" folder
    - Mask images are in the "labelled_images" folder with same naming pattern
    """
    import os
    from glob import glob
    from sklearn.model_selection import train_test_split

    pre_imgs = []
    post_imgs = []
    masks = []

    # Path to images and masks directories
    images_dir = os.path.join(config.DATA_DIR, "images")
    masks_dir = os.path.join(config.DATA_DIR, "labelled_images")

    # Verify directories exist
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not os.path.exists(masks_dir):
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

    print(f"Searching for images in {images_dir}")
    print(f"Searching for masks in {masks_dir}")

    # Get all files in the images directory
    all_image_files = os.listdir(images_dir)

    # Extract pre-disaster images
    pre_image_files = [f for f in all_image_files if "_pre_disaster" in f]

    if not pre_image_files:
        raise FileNotFoundError(
            f"No pre-disaster images found in {images_dir}")

    print(f"Found {len(pre_image_files)} pre-disaster images")

    # For each pre-disaster image, find matching post-disaster image and mask
    valid_triplets = 0
    missing_post = 0
    missing_mask = 0

    for pre_filename in pre_image_files:
        # Get the base name (without the pre_disaster suffix)
        base_name = pre_filename.replace("_pre_disaster", "").split('.')[0]
        file_ext = pre_filename.split('.')[-1]

        # Construct filenames
        pre_img_path = os.path.join(images_dir, pre_filename)
        post_filename = f"{base_name}_post_disaster.{file_ext}"
        post_img_path = os.path.join(images_dir, post_filename)
        # Mask has same name as post image
        post_mask_path = os.path.join(masks_dir, post_filename)

        # Check if both post image and post mask exist
        if os.path.exists(post_img_path):
            if os.path.exists(post_mask_path):
                pre_imgs.append(pre_img_path)
                post_imgs.append(post_img_path)
                masks.append(post_mask_path)
                valid_triplets += 1
            else:
                missing_mask += 1
                if config.VERBOSE:
                    print(f"Warning: Missing mask for {post_filename}")
        else:
            missing_post += 1
            if config.VERBOSE:
                print(f"Warning: Missing post-disaster image for {base_name}")

    print(
        f"Found {valid_triplets} complete triplets (pre-image, post-image, post-mask)")
    print(f"Missing post-disaster images: {missing_post}")
    print(f"Missing mask files: {missing_mask}")

    if valid_triplets == 0:
        raise ValueError(
            "No matching pre/post/mask triplets found. Check file naming patterns.")

    # If separate test data is available, use less data for validation
    validation_split = 0.2  # Default 20% for validation
    if hasattr(config, 'TEST_DATA_DIR') and os.path.exists(config.TEST_DATA_DIR):
        validation_split = 0.1  # Reduced to 10% when test data is available

    # For very small datasets with separate test data, use all data for training
    if valid_triplets < 30 and hasattr(config, 'TEST_DATA_DIR') and os.path.exists(config.TEST_DATA_DIR):
        validation_split = 0  # No validation split

    # Split into train and validation sets
    if validation_split > 0:
        train_pre, valid_pre, train_post, valid_post, train_masks, valid_masks = train_test_split(
            pre_imgs, post_imgs, masks, test_size=validation_split, random_state=42
        )

        print(f"Training set: {len(train_pre)} samples")
        print(f"Validation set: {len(valid_pre)} samples")
    else:
        # Use all data for training
        train_pre, valid_pre = pre_imgs, []
        train_post, valid_post = post_imgs, []
        train_masks, valid_masks = masks, []
        print(
            f"Using all {len(train_pre)} samples for training (no validation split)")

    return train_pre, valid_pre, train_post, valid_post, train_masks, valid_masks

# Prepare test data


def prepare_test_data(config):
    """
    Prepare test data from a directory structure where:
    - Pre and post disaster images are in the "images" folder
    - Mask images (if available) are in the "labelled_images" folder
    """
    import os

    test_pre_imgs = []
    test_post_imgs = []
    test_masks = []

    # Path to images and masks directories
    images_dir = os.path.join(config.TEST_DATA_DIR, "images")
    masks_dir = os.path.join(config.TEST_DATA_DIR, "labelled_images")

    # Verify images directory exists
    if not os.path.exists(images_dir):
        raise FileNotFoundError(
            f"Test images directory not found: {images_dir}")

    print(f"Searching for test images in {images_dir}")

    # Get all files in the images directory
    all_image_files = os.listdir(images_dir)

    # Extract pre-disaster images
    pre_image_files = [f for f in all_image_files if "_pre_disaster" in f]

    if not pre_image_files:
        raise FileNotFoundError(
            f"No pre-disaster test images found in {images_dir}")

    print(f"Found {len(pre_image_files)} pre-disaster test images")

    # Check if masks directory exists (masks may be optional for testing)
    has_masks = os.path.exists(masks_dir)
    if has_masks:
        print(f"Masks directory found at {masks_dir}. Will evaluate accuracy.")
    else:
        print(
            f"No masks directory found at {masks_dir}. Will run inference only.")

    # For each pre-disaster image, find matching post-disaster image and mask
    valid_pairs = 0

    for pre_filename in pre_image_files:
        # Get the base name (without the pre_disaster suffix)
        base_name = pre_filename.replace("_pre_disaster", "").split('.')[0]
        file_ext = pre_filename.split('.')[-1]

        # Construct filenames
        pre_img_path = os.path.join(images_dir, pre_filename)
        post_filename = f"{base_name}_post_disaster.{file_ext}"
        post_img_path = os.path.join(images_dir, post_filename)

        # Check if post image exists
        if os.path.exists(post_img_path):
            test_pre_imgs.append(pre_img_path)
            test_post_imgs.append(post_img_path)

            # Check for mask if masks directory exists
            if has_masks:
                post_mask_path = os.path.join(masks_dir, post_filename)
                if os.path.exists(post_mask_path):
                    test_masks.append(post_mask_path)
                else:
                    # Append None to maintain alignment with image pairs
                    test_masks.append(None)
                    print(
                        f"Warning: Missing mask for test image {post_filename}")

            valid_pairs += 1
        else:
            print(f"Warning: Missing post-disaster test image for {base_name}")

    print(f"Found {valid_pairs} complete test image pairs")

    # Only return masks if they exist for all images
    if has_masks and None not in test_masks:
        print(f"All test images have corresponding masks. Will evaluate accuracy.")
        return test_pre_imgs, test_post_imgs, test_masks
    elif has_masks:
        print(f"Some test images are missing masks. Will perform inference only.")
        return test_pre_imgs, test_post_imgs, None
    else:
        return test_pre_imgs, test_post_imgs, None


# Augmentations optimized for faster training
def get_transforms(phase, config):
    if phase == 'train':
        return A.Compose([
            # Faster version of RandomResizedCrop
            A.OneOf([
                A.RandomCrop(height=config.IMG_SIZE, width=config.IMG_SIZE),
                A.Resize(height=config.IMG_SIZE, width=config.IMG_SIZE),
            ], p=1.0),

            # Group spatial transforms for faster execution
            A.OneOf([
                A.HorizontalFlip(p=0.8),
                A.VerticalFlip(p=0.8),
                A.RandomRotate90(p=0.8),
                A.Transpose(p=0.8),
            ], p=0.8),

            # Limited color augmentations to save processing time
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=1.0),
                A.HueSaturationValue(
                    hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=1.0),
            ], p=0.5),

            # Normalize and convert to tensor (required)
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ], additional_targets={'image2': 'image'})
    else:
        # Simple and fast validation transforms
        return A.Compose([
            A.Resize(config.IMG_SIZE, config.IMG_SIZE),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ], additional_targets={'image2': 'image'})

# Helper functions for small dataset handling


def fix_scheduler_for_small_datasets(config, train_loader):
    """
    Adjust batch size and scheduler parameters for small datasets
    to prevent 'steps_per_epoch=0' error with OneCycleLR
    """
    # Ensure there's at least one batch per epoch
    num_samples = len(train_loader.dataset)

    # If dataset is too small for current batch size
    if num_samples <= config.TRAIN_BATCH_SIZE:
        # Adjust batch size to be at most half the dataset size
        new_batch_size = max(1, num_samples // 2)
        print(f"Dataset is too small for batch size {config.TRAIN_BATCH_SIZE}")
        print(f"Adjusting batch size to {new_batch_size}")
        config.TRAIN_BATCH_SIZE = new_batch_size

        # Ensure gradient accumulation doesn't cause issues
        config.GRADIENT_ACCUMULATION_STEPS = 1
        print("Setting gradient accumulation steps to 1")

        # Recreate train_loader with new batch size
        return True  # Signal that dataloader needs to be recreated

    # Calculate steps per epoch considering gradient accumulation
    steps_per_epoch = len(train_loader) // config.GRADIENT_ACCUMULATION_STEPS

    # If steps per epoch is 0 after gradient accumulation
    if steps_per_epoch == 0:
        # Adjust gradient accumulation to ensure at least one step per epoch
        config.GRADIENT_ACCUMULATION_STEPS = max(1, len(train_loader) // 2)
        steps_per_epoch = len(
            train_loader) // config.GRADIENT_ACCUMULATION_STEPS

        if steps_per_epoch == 0:
            # If still zero, force it to be at least 1
            steps_per_epoch = 1
            print("Warning: Very small dataset. Setting steps_per_epoch=1 for scheduler")

        print(
            f"Adjusted gradient accumulation steps to {config.GRADIENT_ACCUMULATION_STEPS}")
        print(f"Steps per epoch: {steps_per_epoch}")

    return False  # No need to recreate dataloader

# Create appropriate scheduler based on dataset size


def create_scheduler(optimizer, train_loader, config, stage_epochs):
    """
    Create appropriate scheduler based on dataset size - using CosineAnnealingLR for all cases
    to avoid OneCycleLR step counting issues
    """
    # Calculate steps per epoch - ensure it's at least 1
    steps_per_epoch = max(1, len(train_loader) //
                          config.GRADIENT_ACCUMULATION_STEPS)

    # Calculate total steps
    total_steps = steps_per_epoch * stage_epochs

    print(
        f"Creating scheduler with {steps_per_epoch} steps per epoch and {total_steps} total steps")

    # Use CosineAnnealingLR for all datasets to avoid OneCycleLR step counting issues
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-6
    )

    # Return total steps so we can track them in the training loop
    return scheduler, total_steps
# Loss function for segmentation


class CombinedLoss(nn.Module):
    def __init__(self, weights=None, ignore_index=255):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(
            weight=weights, ignore_index=ignore_index)

    def forward(self, pred, target):
        # Cross-entropy loss
        ce = self.ce_loss(pred, target)

        # Dice loss for each class
        dice_loss = 0
        pred_softmax = F.softmax(pred, dim=1)

        for i in range(pred.shape[1]):
            pred_i = pred_softmax[:, i]
            target_i = (target == i).float()

            intersection = (pred_i * target_i).sum()
            dice_coef = (2. * intersection + 1e-6) / \
                (pred_i.sum() + target_i.sum() + 1e-6)
            dice_loss += 1 - dice_coef

        dice_loss /= pred.shape[1]  # Average across classes

        # Combine losses
        total_loss = ce + dice_loss

        return total_loss

# Alpha-blended overlay of damage predictions on post-disaster image


def overlay_damage_mask(post_img, damage_mask, alpha=0.7):
    """
    Create an alpha-blended overlay of the damage mask on the post-disaster image

    Args:
        post_img: Post-disaster image (RGB)
        damage_mask: Damage mask (class indices)
        alpha: Opacity of the overlay

    Returns:
        Overlaid image
    """
    # Updated color map for 5 classes
    color_map = {
        0: [0, 0, 0],        # Background (black)
        1: [255, 255, 255],  # No damage (white)
        2: [0, 255, 0],      # Minor damage (green)
        3: [255, 165, 0],    # Major damage (orange)
        4: [255, 0, 0]       # Destroyed (red)
    }

    # Convert mask to RGB
    mask_rgb = np.zeros(
        (damage_mask.shape[0], damage_mask.shape[1], 3), dtype=np.uint8)

    for class_idx, color in color_map.items():
        if class_idx < len(color_map):  # Ensure the class is in our color map
            mask_rgb[damage_mask == class_idx] = color

    # Only overlay non-background pixels
    overlay = post_img.copy()
    for i in range(damage_mask.shape[0]):
        for j in range(damage_mask.shape[1]):
            if damage_mask[i, j] > 0:  # Non-background
                overlay[i, j] = alpha * mask_rgb[i, j] + \
                    (1 - alpha) * overlay[i, j]

    return overlay


def test_model(model, config):
    """
    Test the model on a separate test dataset
    """
    import os
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    from tqdm import tqdm

    print("Starting model testing on separate test dataset...")

    # Prepare test data
    test_pre_imgs, test_post_imgs, test_masks = prepare_test_data(config)
    has_masks = test_masks is not None

    # Create test results directory if needed
    if config.SAVE_TEST_RESULTS:
        os.makedirs(config.TEST_RESULTS_DIR, exist_ok=True)
        print(f"Test results will be saved to {config.TEST_RESULTS_DIR}")

    # Set model to evaluation mode
    model.eval()

    # Track metrics if masks are available
    if has_masks:
        # Metrics to track (Dice, IoU per class)
        class_dice_scores = [[] for _ in range(config.NUM_CLASSES)]
        class_iou_scores = [[] for _ in range(config.NUM_CLASSES)]

    # Process all test images
    start_time = time.time()

    for idx in tqdm(range(len(test_pre_imgs)), desc="Testing"):
        pre_img_path = test_pre_imgs[idx]
        post_img_path = test_post_imgs[idx]

        # Get file name for saving results
        file_name = os.path.basename(post_img_path).replace(
            "_post_disaster", "").split('.')[0]

        # Run prediction
        damage_mask, overlay = predict_damage(
            model, pre_img_path, post_img_path, config)

        # Save results if enabled
        if config.SAVE_TEST_RESULTS:
            # Create color-coded damage mask
            color_map = {
                0: [0, 0, 0],        # Background (black)
                1: [255, 255, 255],  # No damage (white)
                2: [0, 255, 0],      # Minor damage (green)
                3: [255, 165, 0],    # Major damage (orange)
                4: [255, 0, 0]       # Destroyed (red)
            }

            # Convert to RGB visualization
            mask_rgb = np.zeros(
                (damage_mask.shape[0], damage_mask.shape[1], 3), dtype=np.uint8)
            for class_idx, color in color_map.items():
                if class_idx < config.NUM_CLASSES:  # Ensure the class is in our color map
                    mask_rgb[damage_mask == class_idx] = color

            # Save mask and overlay
            Image.fromarray(mask_rgb).save(os.path.join(
                config.TEST_RESULTS_DIR, f"{file_name}_mask.png"))
            Image.fromarray(overlay).save(os.path.join(
                config.TEST_RESULTS_DIR, f"{file_name}_overlay.png"))

            # If available, save side-by-side comparison with ground truth
            if has_masks and test_masks[idx] is not None:
                mask_path = test_masks[idx]
                true_mask = np.array(Image.open(mask_path))

                # Convert RGB mask to class indices if needed
                if len(true_mask.shape) == 3 and true_mask.shape[2] == 3:
                    # Convert using the same logic as in DisasterDataset
                    temp_mask = np.zeros(true_mask.shape[:2], dtype=np.uint8)

                    # Class 0: Background (black)
                    temp_mask[(true_mask[:, :, 0] < 10) & (
                        true_mask[:, :, 1] < 10) & (true_mask[:, :, 2] < 10)] = 0

                    # Class 1: Undamaged (white)
                    temp_mask[(true_mask[:, :, 0] > 200) & (
                        true_mask[:, :, 1] > 200) & (true_mask[:, :, 2] > 200)] = 1

                    # Class 2: Minor damage (green)
                    temp_mask[(true_mask[:, :, 0] < 100) & (
                        true_mask[:, :, 1] > 200) & (true_mask[:, :, 2] < 100)] = 2

                    # Class 3: Major damage (orange)
                    temp_mask[(true_mask[:, :, 0] > 200) & (true_mask[:, :, 1] > 100) & (
                        true_mask[:, :, 1] < 200) & (true_mask[:, :, 2] < 100)] = 3

                    # Class 4: Destroyed (red)
                    temp_mask[(true_mask[:, :, 0] > 200) & (
                        true_mask[:, :, 1] < 100) & (true_mask[:, :, 2] < 100)] = 4

                    true_mask = temp_mask

                # Load post image for visualization
                post_img = np.array(Image.open(post_img_path).convert('RGB'))

                # Resize true_mask to match prediction size if they don't match
                if true_mask.shape != damage_mask.shape:
                    print(
                        f"Resizing ground truth from {true_mask.shape} to {damage_mask.shape}")
                    # Use nearest neighbor interpolation to preserve class indices
                    true_mask = cv2.resize(true_mask, (damage_mask.shape[1], damage_mask.shape[0]),
                                           interpolation=cv2.INTER_NEAREST)

                # Resize post image to match prediction size if needed
                if post_img.shape[:2] != damage_mask.shape:
                    post_img = cv2.resize(
                        post_img, (damage_mask.shape[1], damage_mask.shape[0]))

                # Calculate metrics
                for class_idx in range(config.NUM_CLASSES):
                    # Calculate Dice coefficient
                    pred_class = (damage_mask == class_idx).astype(np.float32)
                    true_class = (true_mask == class_idx).astype(np.float32)

                    intersection = np.sum(pred_class * true_class)
                    dice = (2. * intersection + 1e-6) / \
                        (np.sum(pred_class) + np.sum(true_class) + 1e-6)
                    class_dice_scores[class_idx].append(dice)

                    # Calculate IoU
                    union = np.sum(pred_class) + \
                        np.sum(true_class) - intersection
                    iou = (intersection + 1e-6) / (union + 1e-6)
                    class_iou_scores[class_idx].append(iou)

                # Create true mask RGB visualization
                true_mask_rgb = np.zeros(
                    (true_mask.shape[0], true_mask.shape[1], 3), dtype=np.uint8)
                for class_idx, color in color_map.items():
                    if class_idx < config.NUM_CLASSES:  # Ensure the class is in our color map
                        true_mask_rgb[true_mask == class_idx] = color

                # Create visualization
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))

                axes[0].imshow(post_img)
                axes[0].set_title('Post-Disaster Image')
                axes[0].axis('off')

                axes[1].imshow(true_mask_rgb)
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')

                axes[2].imshow(mask_rgb)
                axes[2].set_title('Prediction')
                axes[2].axis('off')

                axes[3].imshow(overlay)
                axes[3].set_title('Overlay')
                axes[3].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(config.TEST_RESULTS_DIR,
                            f"{file_name}_comparison.png"))
                plt.close()

    # Calculate overall metrics if ground truth is available
    if has_masks:
        # Calculate mean metrics
        mean_dice = []
        mean_iou = []

        for class_idx in range(config.NUM_CLASSES):
            if class_dice_scores[class_idx]:  # If the class exists in the dataset
                mean_dice.append(np.mean(class_dice_scores[class_idx]))
                mean_iou.append(np.mean(class_iou_scores[class_idx]))

                print(
                    f"Class {class_idx} - Dice: {mean_dice[-1]:.4f}, IoU: {mean_iou[-1]:.4f}")

        # Overall metrics
        overall_dice = np.mean(mean_dice)
        overall_iou = np.mean(mean_iou)

        print(
            f"Overall Metrics - Dice: {overall_dice:.4f}, IoU: {overall_iou:.4f}")

        # Save metrics to a file
        with open(os.path.join(config.TEST_RESULTS_DIR, "metrics.txt"), "w") as f:
            f.write(f"Overall Dice: {overall_dice:.4f}\n")
            f.write(f"Overall IoU: {overall_iou:.4f}\n\n")

            for class_idx in range(config.NUM_CLASSES):
                if class_dice_scores[class_idx]:
                    f.write(
                        f"Class {class_idx} - Dice: {mean_dice[class_idx]:.4f}, IoU: {mean_iou[class_idx]:.4f}\n")

    total_time = time.time() - start_time
    print(f"Testing completed in {total_time:.2f} seconds")
    print(f"Processed {len(test_pre_imgs)} test images")

    if config.SAVE_TEST_RESULTS:
        print(f"Test results saved to {config.TEST_RESULTS_DIR}")

# Function to predict damage on new images


def predict_damage(model, pre_img_path, post_img_path, config, overlay_alpha=0.7):
    """
    Function to predict damage on a single pre/post image pair

    Args:
        model: Trained model
        pre_img_path: Pre-disaster image path
        post_img_path: Post-disaster image path
        config: Configuration object
        overlay_alpha: Alpha value for blending the damage mask overlay

    Returns:
        Damage segmentation mask and overlaid visualization
    """
    model.eval()

    transform = get_transforms('valid', config)

    pre_img = np.array(Image.open(pre_img_path).convert('RGB'))
    post_img = np.array(Image.open(post_img_path).convert('RGB'))

    # Original post image for visualization (before normalization)
    original_post = post_img.copy()

    augmented = transform(image=pre_img, image2=post_img)
    pre_tensor = augmented['image'].unsqueeze(0)
    post_tensor = augmented['image2'].unsqueeze(0)

    with torch.no_grad():
        # Fix for deprecation warning
        if config.USE_AMP:
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                output = model(pre_tensor.to(config.DEVICE),
                               post_tensor.to(config.DEVICE))
        else:
            output = model(pre_tensor.to(config.DEVICE),
                           post_tensor.to(config.DEVICE))

    # Get predicted mask
    pred_mask = torch.argmax(output, dim=1).cpu().numpy()[0]

    # Resize original post image and predicted mask to same size if needed
    if original_post.shape[:2] != pred_mask.shape:
        original_post = cv2.resize(
            original_post, (pred_mask.shape[1], pred_mask.shape[0]))

    # Create overlay
    overlay = overlay_damage_mask(original_post, pred_mask, overlay_alpha)

    return pred_mask, overlay

# Training function with gradient accumulation and time tracking


# Updated training function with fixes for both deprecation warnings and scheduler issues
def train_fn(dataloader, model, criterion, optimizer, device, scheduler_info=None, config=None):
    model.train()

    # Unpack scheduler and total steps
    scheduler, max_steps = scheduler_info if scheduler_info else (None, 0)
    current_steps = 0  # Initialize step counter

    # Fix for GradScaler deprecation warning
    if config.USE_AMP:
        scaler = torch.amp.GradScaler()
    else:
        scaler = None

    train_loss = 0.0
    dataset_size = 0

    # Track batch processing time
    from time import time
    start_time = time()

    progress_bar = tqdm(dataloader, desc='Training')

    # For gradient accumulation
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(progress_bar):
        pre_images = batch['pre_image'].to(device, non_blocking=True)
        post_images = batch['post_image'].to(device, non_blocking=True)
        masks = batch['mask'].long().to(device, non_blocking=True)
        batch_size = pre_images.shape[0]

        # Forward pass with mixed precision
        if config.USE_AMP:
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(pre_images, post_images)
                loss = criterion(outputs, masks)
                loss = loss / config.GRADIENT_ACCUMULATION_STEPS
        else:
            outputs = model(pre_images, post_images)
            loss = criterion(outputs, masks)
            loss = loss / config.GRADIENT_ACCUMULATION_STEPS

        # Backward pass
        if config.USE_AMP:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update weights after accumulating gradients
        if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
            # Unscale gradients for proper gradient clipping
            if config.USE_AMP:
                scaler.unscale_(optimizer)

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step
            if config.USE_AMP:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

            # Learning rate scheduler step - only if we haven't reached max steps
            if scheduler is not None and current_steps < max_steps:
                scheduler.step()
                current_steps += 1

                # Log every 100 steps
                if current_steps % 100 == 0:
                    print(f"Scheduler step {current_steps}/{max_steps}")

        # Track loss
        train_loss += (loss.item() *
                       config.GRADIENT_ACCUMULATION_STEPS) * batch_size
        dataset_size += batch_size

        # Show progress
        progress_bar.set_postfix(
            loss=train_loss/dataset_size,
            steps=f"{current_steps}/{max_steps}"
        )

    # Handle case where last batch didn't trigger parameter update
    if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS != 0:
        if config.USE_AMP:
            scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if config.USE_AMP:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad()

        # DO NOT step scheduler here

    train_epoch_loss = train_loss / dataset_size
    print(f"Epoch completed. Scheduler steps: {current_steps}/{max_steps}")

    return train_epoch_loss
# Validation function


def valid_fn(dataloader, model, criterion, device, config=None):
    model.eval()

    valid_loss = 0.0
    dataset_size = 0

    # Skip validation for very small datasets
    if len(dataloader.dataset) <= 1:
        print("Warning: Validation dataset too small. Skipping validation.")
        return 0.0

    progress_bar = tqdm(dataloader, desc='Validation')

    with torch.no_grad():
        for batch in progress_bar:
            pre_images = batch['pre_image'].to(device, non_blocking=True)
            post_images = batch['post_image'].to(device, non_blocking=True)
            masks = batch['mask'].long().to(
                device, non_blocking=True)  # Ensure Long type
            batch_size = pre_images.shape[0]

            with torch.cuda.amp.autocast(enabled=config.USE_AMP):
                outputs = model(pre_images, post_images)
                loss = criterion(outputs, masks)

            valid_loss += loss.item() * batch_size
            dataset_size += batch_size

            progress_bar.set_postfix(loss=valid_loss/dataset_size)

    valid_epoch_loss = valid_loss / dataset_size if dataset_size > 0 else 0

    return valid_epoch_loss

# Run training with checkpointing and small dataset support


def run_training(config):
    import time

    start_time = time.time()
    print(f"Starting training with target completion time under 18 hours")

    # Prepare data
    train_pre, valid_pre, train_post, valid_post, train_masks, valid_masks = prepare_data(
        config)

    # Check if validation set exists
    has_validation = len(valid_pre) > 0

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        save_dir='checkpoints', model_name='siamese_unet')

    # Set up progressive resizing if enabled
    if config.PROGRESSIVE_RESIZING:
        current_img_size = config.INITIAL_IMG_SIZE
        print(
            f"Starting with progressive resizing: Initial size = {current_img_size}px")
    else:
        current_img_size = config.IMG_SIZE

    # Track total time to monitor 18-hour limit
    total_time_limit = 18 * 3600  # 18 hours in seconds

    # Training stages for progressive resizing
    stages = []
    if config.PROGRESSIVE_RESIZING:
        stages = [
            {"size": config.INITIAL_IMG_SIZE, "epochs": config.PROGRESSIVE_EPOCHS},
            {"size": config.FINAL_IMG_SIZE,
                "epochs": config.EPOCHS - config.PROGRESSIVE_EPOCHS}
        ]
    else:
        stages = [{"size": config.IMG_SIZE, "epochs": config.EPOCHS}]

    # Initialize model
    model = SiameseUNet(config)
    model.to(config.DEVICE)

    # Enable benchmark mode for faster training on fixed size inputs
    torch.backends.cudnn.benchmark = True

    # Define loss function with class weights
    if config.CLASS_WEIGHTS:
        weights = torch.tensor(config.CLASS_WEIGHTS).float().to(config.DEVICE)
        print(f"Using class weights: {weights}")
    else:
        weights = None
        print("Using no class weights")

    criterion = CombinedLoss(weights=weights)

    # Variables to track training progress
    best_valid_loss = float('inf')
    early_stopping_counter = 0
    global_epoch = 0
    current_stage_idx = 0
    current_stage_epoch = 0

    # Try to load checkpoint if exists
    checkpoint = checkpoint_manager.load_latest_checkpoint(model)

    if checkpoint:
        # Resume from checkpoint
        global_epoch = checkpoint['epoch']
        current_stage_idx = checkpoint['stage_idx']
        current_stage_epoch = checkpoint['curr_stage_epoch']
        best_valid_loss = checkpoint['best_valid_loss']
        early_stopping_counter = checkpoint['early_stopping_counter']

        # Restore config from checkpoint if needed
        for key, value in checkpoint['config'].items():
            if hasattr(config, key):
                setattr(config, key, value)

        print(
            f"Resuming training from epoch {global_epoch}, stage {current_stage_idx}, stage epoch {current_stage_epoch}")
        print(f"Best validation loss so far: {best_valid_loss}")

    # Training loop with progressive resizing
    for stage_idx in range(current_stage_idx, len(stages)):
        stage = stages[stage_idx]
        current_img_size = stage["size"]
        stage_epochs = stage["epochs"]

        print(
            f"Starting/Resuming training stage {stage_idx+1}/{len(stages)} with image size {current_img_size}px")

        # Update config for current stage
        config.IMG_SIZE = current_img_size

        # Create datasets and dataloaders with current image size
        train_dataset = DisasterDataset(
            pre_imgs=train_pre,
            post_imgs=train_post,
            masks=train_masks,
            transform=get_transforms('train', config)
        )

        valid_dataset = None
        valid_loader = None
        if has_validation:
            valid_dataset = DisasterDataset(
                pre_imgs=valid_pre,
                post_imgs=valid_post,
                masks=valid_masks,
                transform=get_transforms('valid', config)
            )

            valid_loader = DataLoader(
                valid_dataset,
                batch_size=config.VALID_BATCH_SIZE,
                shuffle=False,
                num_workers=min(4, os.cpu_count() or 1),
                pin_memory=True,
                prefetch_factor=2
            )

        # Use optimal workers for dataloading
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=min(4, os.cpu_count() or 1),
            pin_memory=True,
            drop_last=False,  # Changed to False for small datasets
            prefetch_factor=2
        )

        # Check and adjust for small dataset size
        recreate_loader = fix_scheduler_for_small_datasets(
            config, train_loader)

        # If batch size was adjusted, recreate the dataloader
        if recreate_loader:
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.TRAIN_BATCH_SIZE,
                shuffle=True,
                num_workers=min(4, os.cpu_count() or 1),
                pin_memory=True,
                drop_last=False,
                prefetch_factor=2
            )

            # Re-check after recreation to ensure we have at least one batch
            if len(train_loader) == 0:
                print(f"Error: Dataset too small even after batch size adjustment")
                print(
                    f"Dataset size: {len(train_dataset)}, Batch size: {config.TRAIN_BATCH_SIZE}")
                raise ValueError(
                    "Dataset too small for training. Need at least 2 samples.")

        # Optimizer with learning rate scaled for image size
        lr_scale = current_img_size / config.FINAL_IMG_SIZE
        adjusted_lr = config.LEARNING_RATE * lr_scale

        optimizer = optim.AdamW(
            model.parameters(),
            lr=adjusted_lr,
            weight_decay=config.WEIGHT_DECAY
        )

        # Create appropriate scheduler based on dataset size - returns (scheduler, max_steps)
        scheduler_info = create_scheduler(
            optimizer, train_loader, config, stage_epochs)

        # Extract scheduler from scheduler_info for use in checkpoint loading/saving
        scheduler, max_scheduler_steps = scheduler_info

        print(f"Created scheduler with maximum {max_scheduler_steps} steps")

        # If resuming, load optimizer and scheduler states
        if checkpoint and stage_idx == current_stage_idx:
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Optimizer state loaded")

            if checkpoint.get('scheduler_state_dict'):
                try:
                    scheduler.load_state_dict(
                        checkpoint['scheduler_state_dict'])
                    print("Scheduler state loaded")
                except Exception as e:
                    print(
                        f"Could not load scheduler state - creating new scheduler: {e}")
                    # Recreate scheduler_info since we couldn't load the state
                    scheduler_info = create_scheduler(
                        optimizer, train_loader, config, stage_epochs)
                    scheduler, max_scheduler_steps = scheduler_info
                    print(
                        f"New scheduler created with {max_scheduler_steps} steps")

        # Training loop for current stage - start from the current stage epoch if resuming
        start_epoch = current_stage_epoch if stage_idx == current_stage_idx else 0

        for epoch in range(start_epoch, stage_epochs):
            # Check remaining time to ensure we don't exceed 18 hours
            elapsed_time = time.time() - start_time
            if elapsed_time > total_time_limit * 0.9:  # 90% of time limit
                print(
                    f"Warning: Approaching 18-hour limit. Training will be terminated early.")
                print(f"Elapsed time: {elapsed_time/3600:.2f} hours")

                # Save final checkpoint and exit - use the extracted scheduler
                checkpoint_manager.save_checkpoint(
                    model, optimizer, scheduler, global_epoch, stage_idx, epoch,
                    best_valid_loss, early_stopping_counter, config, stage
                )

                # Save current best model
                torch.save(model.state_dict(), 'final_siamese_unet_model.pth')
                return model

            # Update global and stage-specific epoch counters
            current_stage_epoch = epoch
            global_epoch += 1

            print(
                f'Stage {stage_idx+1}/{len(stages)}, Epoch {epoch+1}/{stage_epochs}, Global epoch {global_epoch}')
            print(f'Elapsed time: {elapsed_time/3600:.2f} hours')

            # Training - pass the full scheduler_info to train_fn
            train_loss = train_fn(
                train_loader, model, criterion, optimizer,
                config.DEVICE, scheduler_info, config
            )

            # Validation - only if validation set exists
            if has_validation:
                valid_loss = valid_fn(
                    valid_loader, model, criterion,
                    config.DEVICE, config
                )

                print(
                    f'Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}')

                # Save model if validation loss improves
                if valid_loss < best_valid_loss:
                    print(
                        f'Validation Loss Improved ({best_valid_loss:.4f} -> {valid_loss:.4f})')
                    best_valid_loss = valid_loss

                    # Save best model
                    torch.save(model.state_dict(),
                               'best_siamese_unet_model.pth')
                    # Also save as a specifically named checkpoint
                    torch.save(model.state_dict(), checkpoint_manager.save_dir /
                               f"{checkpoint_manager.model_name}_best.pth")

                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    print(
                        f'EarlyStopping counter: {early_stopping_counter} out of {config.EARLY_STOPPING_PATIENCE}')

                    if early_stopping_counter >= config.EARLY_STOPPING_PATIENCE:
                        print(
                            f'Early stopping at stage {stage_idx+1}, epoch {epoch+1}!')
                        # Continue to next stage if there is one
                        break
            else:
                # No validation set - save model periodically instead
                print(f'Train Loss: {train_loss:.4f} (No validation set)')
                if epoch % 5 == 0 or epoch == stage_epochs - 1:
                    print(
                        f'Saving model at epoch {epoch+1} (no validation set)')
                    torch.save(model.state_dict(),
                               'best_siamese_unet_model.pth')
                    torch.save(model.state_dict(), checkpoint_manager.save_dir /
                               f"{checkpoint_manager.model_name}_best.pth")

            # Save checkpoint periodically to prevent data loss - use the extracted scheduler
            if epoch % config.CHECKPOINT_FREQ == 0:
                checkpoint_manager.save_checkpoint(
                    model, optimizer, scheduler, global_epoch, stage_idx, epoch,
                    best_valid_loss, early_stopping_counter, config, stage
                )

    # Final timing report
    total_time = time.time() - start_time
    hours = total_time / 3600
    print(f"Total training time: {hours:.2f} hours")

    if hours < 18:
        print(
            f"Successfully completed training within 18-hour limit! ({hours:.2f} hours)")
    else:
        print(
            f"Warning: Training exceeded 18-hour target! ({hours:.2f} hours)")

    return model
# Complete main function with checkpoint handling and test dataset support


def main():
    # Check for existing training
    checkpoint_manager = CheckpointManager()

    # Create config object
    config = Config()

    # Define mode: 'train', 'test', or 'both'
    try:
        mode = input(
            "Select mode ('train', 'test', or 'both'): ").lower().strip()
        if mode not in ['train', 'test', 'both']:
            mode = 'both'  # Default to both if invalid input
    except:
        # If running in non-interactive environment, default to both
        mode = 'both'
        print("Running in non-interactive mode. Will run both training and testing.")

    # Train if requested
    if mode in ['train', 'both']:
        # Check if a previous training session exists
        previous_training_exists = checkpoint_manager.metadata_path.exists()

        if previous_training_exists:
            try:
                # If running in interactive environment, ask user
                resume = input(
                    "Previous training checkpoint found. Resume training? (y/n): ").lower() == 'y'
            except:
                # If running in a non-interactive environment, default to resuming
                resume = True
                print(
                    "Running in non-interactive mode. Automatically resuming previous training.")
        else:
            resume = False

        if resume:
            # Load the latest checkpoint
            checkpoint = checkpoint_manager.load_latest_checkpoint()

            # Restore config from checkpoint if available
            if checkpoint and 'config' in checkpoint:
                for key, value in checkpoint['config'].items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                print("Configuration restored from checkpoint")

        # Run or resume training
        model = run_training(config)

        # Load best model for inference
        best_model_path = checkpoint_manager.find_best_model()
        if best_model_path:
            model.load_state_dict(torch.load(best_model_path))
            print(f"Loaded best model from {best_model_path}")
        else:
            print("Using final model for inference")

    # Test mode only - load model without training
    elif mode == 'test':
        # Initialize model
        model = SiameseUNet(config)
        model.to(config.DEVICE)

        # Try to find the best model
        best_model_path = checkpoint_manager.find_best_model()

        if best_model_path:
            print(f"Loading best model from {best_model_path}")
            model.load_state_dict(torch.load(best_model_path))
        else:
            # Look for any saved model
            model_path = 'best_siamese_unet_model.pth'
            if os.path.exists(model_path):
                print(f"Loading model from {model_path}")
                model.load_state_dict(torch.load(model_path))
            else:
                print("No trained model found. Please train the model first.")
                return

    # Run testing if in test or both modes
    has_separate_test = hasattr(
        config, 'TEST_DATA_DIR') and os.path.exists(config.TEST_DATA_DIR)
    if mode in ['test', 'both'] and has_separate_test:
        test_model(model, config)
    elif mode == 'test' and not has_separate_test:
        print("No separate test dataset found. Please specify TEST_DATA_DIR in config.")


if __name__ == "__main__":
    main()
