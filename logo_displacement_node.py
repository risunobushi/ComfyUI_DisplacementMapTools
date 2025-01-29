import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class DisplaceLogo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "logo": ("IMAGE",),  # Real logo image
                "displacement_map": ("IMAGE",),  # Displacement map
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),  # Displacement scale
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("displaced_logo",)
    FUNCTION = "displace_logo"
    CATEGORY = "image/processing"
    OUTPUT_NODE = False

    def displace_logo(self, logo, displacement_map, scale):
        # Convert ComfyUI image tensors to numpy arrays
        if isinstance(logo, torch.Tensor):
            logo = logo.cpu().numpy()
        if isinstance(displacement_map, torch.Tensor):
            displacement_map = displacement_map.cpu().numpy()

        # Remove batch dimensions if present
        if logo.ndim == 4:
            logo = logo[0]  # Remove batch dimension
        if displacement_map.ndim == 4:
            displacement_map = displacement_map[0]  # Remove batch dimension

        # Ensure the displacement map is single-channel (grayscale)
        if displacement_map.shape[0] == 3:
            displacement_map = np.mean(displacement_map, axis=0)  # Convert to grayscale
        else:
            displacement_map = displacement_map[0]  # Use the first channel

        # Resize the displacement map to match the logo's dimensions
        height, width, channels = logo.shape
        displacement_map = np.array(Image.fromarray(displacement_map).resize((width, height), Image.BILINEAR))

        # Scale displacement map
        displacement_map = displacement_map * scale

        # Create grid for displacement
        x, y = np.meshgrid(np.arange(width), np.arange(height))

        # Apply displacement
        new_x = x + displacement_map * (width / height)  # Scale with aspect ratio
        new_y = y + displacement_map

        # Normalize new_x and new_y to [-1, 1] for grid_sample
        new_x = new_x / (width - 1) * 2 - 1
        new_y = new_y / (height - 1) * 2 - 1

        # Stack to create grid tensor
        grid = torch.stack([torch.tensor(new_x, dtype=torch.float32), torch.tensor(new_y, dtype=torch.float32)], dim=-1).unsqueeze(0)

        # Convert logo to tensor and apply grid_sample
        logo_tensor = torch.from_numpy(logo).permute(2, 0, 1).unsqueeze(0).float()
        displaced_logo = F.grid_sample(logo_tensor, grid, mode='bilinear', padding_mode='border', align_corners=True)

        # Convert back to ComfyUI format
        displaced_logo = displaced_logo.squeeze(0).permute(1, 2, 0).unsqueeze(0)

        return (displaced_logo,)

# Node mappings
NODE_CLASS_MAPPINGS = {
    "DisplaceLogo": DisplaceLogo
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DisplaceLogo": "Displace Logo"
}
