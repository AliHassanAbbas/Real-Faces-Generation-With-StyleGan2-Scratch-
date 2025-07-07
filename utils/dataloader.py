import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CelebADataset(Dataset):
    """
    PyTorch Dataset for CelebA images with professional error handling
    and consistent preprocessing for GAN training.
    """
    def __init__(self, root_dir, image_size=64):
        """
        Args:
            root_dir (str): Directory containing CelebA images.
            image_size (int): Target image size after resizing.
        """
        super().__init__()
        self.root_dir = root_dir
        self.image_paths = self._load_image_paths()
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),  # Normalize to [-1, 1]
        ])

    def _load_image_paths(self):
        """
        Returns:
            List of valid image paths in root_dir, sorted for reproducibility.
        Raises:
            RuntimeError if no images found.
        """
        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")
        
        image_paths = [
            os.path.join(self.root_dir, fname)
            for fname in sorted(os.listdir(self.root_dir))
            if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        if not image_paths:
            raise RuntimeError(f"No images found in directory: {self.root_dir}")
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Loads and transforms the image at the given index.
        
        Returns:
            Tensor: Normalized image tensor of shape (3, H, W).
        """
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise IOError(f"Error loading image {img_path}: {e}")
        return self.transform(image)
