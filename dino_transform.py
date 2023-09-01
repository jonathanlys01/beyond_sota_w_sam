import torch
from torchvision.transforms import functional as F

from typing import Optional, Union, Sequence, List, Tuple
import warnings
import math


class DiNORandomResizedCrop(torch.nn.Module):
      
    """
    Similar to the random resized crop of torchvision, but the distribution 
    of the crop's center is DiNOv2's self attention distribution.

    Should be placed after the Normalize and ToTensor transforms, so that the image is suitable for DiNOv2 (there are no preprocessing transforms)
    """

    def __init__(
    self,
    size,
    dino_type: str,
    scale=(0.08, 1.0),
    ratio=(3.0 / 4.0, 4.0 / 3.0),
    temperature: float = 10,
    kernel_size: int = 5,
    interpolation=F.InterpolationMode.BILINEAR,
    antialias: Optional[Union[str, bool]] = "warn",
    ):
        super().__init__()

        self.size = size

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        self.interpolation = interpolation
        self.antialias = antialias
        self.scale = scale
        self.ratio = ratio

        dino_types = ['dinov2_vits14','dinov2_vitb14','dinov2_vitl14','dinov2_vitg14']
        assert dino_type in dino_types, f"dino_type should be one of {dino_types}, got {dino_type}"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading DiNOv2 model {dino_type} on device {self.device}")
        self.model = torch.hub.load('facebookresearch/dinov2', dino_type, pretrained=True).to(self.device)

        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()

        self.density_generator = DensityGenerator(temperature=temperature,
                                                  kernel_size=kernel_size,
                                                  ).to(self.device)

    def get_map(self, img: torch.Tensor) -> torch.Tensor:
        _, height, width = F.get_dimensions(img)
        img = F.resize(img, (height - (height % 14), width - (width % 14)), interpolation=F.InterpolationMode.BICUBIC, antialias=self.antialias)
        _, height, width = F.get_dimensions(img)

        img = img.to(self.device)
        attention_map = get_self_attention(self.model, img) # (num_heads, H//14, W//14)
        density_map = self.density_generator(attention_map).cpu()
        density_map = F.resize(density_map.unsqueeze(0), (height, width), interpolation=F.InterpolationMode.BICUBIC, antialias=self.antialias).squeeze() # (H, W)

        density_map = density_map / torch.sum(density_map)

        return density_map


    def get_params(self, img: torch.Tensor, scale: List[float], ratio: List[float]) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """

        ########  Resize for DiNOv2 (so that the image size is divisible by 14)  ######################


        _, height, width = F.get_dimensions(img)

        area = height * width

        # Get distribution of the attention map

        img = img.to(self.device)
        attention_map = get_self_attention(self.model, img) # (num_heads, H//14, W//14)


        # Generate density map

        density_map = self.density_generator(attention_map) # (H//14, W//14)

        density_map = F.resize(density_map.unsqueeze(0), (height, width), interpolation=F.InterpolationMode.BICUBIC, antialias=self.antialias).squeeze() # (H, W)

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))


            if 0 < w <= width and 0 < h <= height:

                """i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()"""

                x,y = self.generate_point(density_map, (h,w))

                i = y - h // 2
                j = x - w // 2
            
                return i, j, h, w


        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation, antialias=self.antialias)



    def generate_point(
            self,
                    density:torch.Tensor, 
                   target_size: tuple
                   ) -> tuple [int, int]:

        """
        Generate a point from a density map

        Args:
            density: A tensor of shape (H, W)
            og_size: The size of the original image (H, W)
        Returns:
            A tensor of shape (2,) containing the coordinates of the point

        """

        H, W = density.shape

        h,w = target_size


        density = density.flatten()

        density = density / torch.sum(density)

        cum_density = torch.cumsum(density, dim=0)


        u = torch.rand(1).to(self.device)
        # the cumulative density is sorted in ascending order, so we find the nearest value 
        i = torch.searchsorted(cum_density, u).cpu()
        # we get the coordinates of the point
        x = i % W
        y = i // W


        # the crop must be within the image
        for _ in range(10):

            if (y - h // 2 < 0 or y + h // 2 >= H or x - w // 2 < 0 or x + w // 2 >= W):
                u = torch.rand(1).to(self.device)
                i = torch.searchsorted(cum_density, u).cpu()
                x = i % W
                y = i // W
            else : return x,y # int, int

        # if we can't find a point, we return the center of the image (fallback to center crop)
        return W//2, H//2


    def __repr__(self) -> str:
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + f"(size={self.size}"
        format_string += f", scale={tuple(round(s, 4) for s in self.scale)}"
        format_string += f", ratio={tuple(round(r, 4) for r in self.ratio)}"
        format_string += f", interpolation={interpolate_str}"
        format_string += f", antialias={self.antialias})"
        return format_string


def get_self_attention(model,imgs):

    """
    Extract the self attention map from a ViT model (for now, only spports DINOv2)
    
    Args:
        model: A ViT model
        imgs: A tensor of shape (B, 3, H, W) B = 1 
    Returns:
        A tensor of shape (num_heads, H_new, W_new) where H_new and W_new are the height and width of the attention map (size of the img divided by the patch size), CUDA tensor
    """

    if len(imgs.shape) == 3:
        imgs = imgs.unsqueeze(0)
    elif len(imgs.shape) == 4:
        pass
    else:
        raise ValueError("imgs must be a 3D or 4D tensor")

    B, C, H, W = imgs.shape
    
    assert imgs.shape[1] == 3, "imgs must have 3 channels"
    assert imgs.shape[2] % model.patch_size == 0 and imgs.shape[3] % model.patch_size == 0, "sides must be divisible by 14"
    assert B == 1, "B must be 1"

    w_featmap = W // 14
    h_featmap = H // 14

    with torch.no_grad():
        output = model.get_intermediate_layers(x=imgs,
                                            reshape=True,
                                            n = 2,
                                            return_class_token=True,
                                            )

        # output is a list of tuples (maps, class_token), the length of the list is the number of layers considered


        maps = output[0][0] 

        B, C = output[0][1].shape

        # reshape maps to be (B, N, C) where N is the number of patches

        maps = maps.reshape((B,maps.shape[1],-1)).permute(0,2,1)


        class_token = output[0][1].reshape((B,-1,1)).permute(0,2,1)


        maps = torch.cat((class_token, maps), dim=1)

        # get the last attention block (only qkv) with 
    
        qkv = model.blocks[-1].attn.qkv


        B, N, C = maps.shape

        qkv_out = qkv(maps).reshape(B, N, 3, model.num_heads, C // model.num_heads).permute(2, 0, 3, 1, 4) # (3, B, num_heads, N, C//num_heads)


        head_dim = C // model.num_heads
        scale = head_dim**-0.5

        q, k = qkv_out[0] * scale, qkv_out[1]

        attn = q @ k.transpose(-2, -1) # (B, nh, N, N)
        
        nh = model.num_heads
        attn = attn[:, :, 0, 1:].reshape(nh, h_featmap, w_featmap)
        
        return attn


class DensityGenerator(torch.nn.Module):
    """
    Postprocess the attention maps to generate a density map : 
    - Average over the heads
    - Outlier removal (remove the max and replace it by the mean)
    - Apply a gaussian filter
    - Aplly a min max normalization
    - Apply a softmax


    Args:
        kernel_size: The size of the gaussian kernel (Note that all images should be the same size, and are square)
        temperature: The temperature used to apply the softmax
    """

    def __init__(self, kernel_size: int = 3, temperature: float = 0.1):
        super().__init__()
        self.kernel_size = kernel_size
        self.temperature = temperature

        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding="same", bias=False) # padding to keep the same size
        self.conv.weight.data = torch.ones(1, 1, kernel_size, kernel_size)

        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, maps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            maps: A tensor of shape (num_heads, H, W)
        Returns:
            A tensor of shape (H, W)
        """

        maps = torch.mean(maps, dim=0) # average over the heads # (H, W)

        maps = self.conv(maps.unsqueeze(0)) # gaussian filter # (1, H, W)

        # positive values

        maps = (maps - torch.min(maps))
        
        temp = maps.clone()

        temp = self.softmax(temp.flatten()/self.temperature)

        maps = temp.reshape(maps.shape)

        return maps.squeeze(0) # (H, W)