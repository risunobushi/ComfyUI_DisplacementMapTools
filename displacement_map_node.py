import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class ExtractDisplacementMap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # Input image (e.g., normal or bump map)
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),  # Displacement intensity
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("displacement_map",)
    FUNCTION = "extract_displacement_map"
    CATEGORY = "image/processing"
    OUTPUT_NODE = False

    def extract_displacement_map(self, image, intensity):
        # Convert ComfyUI image tensor to numpy array
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if image.ndim == 4:
            image = image[0]  # Remove batch dimension if present
        if image.shape[0] in [1, 3]:
            image = np.transpose(image, (1, 2, 0))  # Convert (C, H, W) to (H, W, C)
        image = (image * 255).astype(np.uint8)  # Scale to 0-255

        # Convert to grayscale
        if image.shape[-1] == 3:
            image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])  # RGB to grayscale
        else:
            image = image[..., 0]  # Use the first channel if already grayscale

        # Normalize and scale by intensity
        displacement_map = image.astype(np.float32) / 255.0
        displacement_map = displacement_map * intensity

        # Convert to tensor and add batch dimension
        displacement_map = torch.from_numpy(displacement_map).unsqueeze(0).unsqueeze(0)
        displacement_map = displacement_map.to(torch.float32)

        return (displacement_map,)

# Node mappings
NODE_CLASS_MAPPINGS = {
    "ExtractDisplacementMap": ExtractDisplacementMap
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExtractDisplacementMap": "Extract Displacement Map"
}
