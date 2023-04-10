import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.ops import roi_pool

# Convolution block for UNet Encoder
class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3,
                      padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3,
                      padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3,
                      padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)


# UNet Encoder
class Unet_encoder(nn.Module):

    def __init__(self,
                 in_channels: int = 3,
                 channels: int = 32,
                 num_pool_layers: int = 4,
                 drop_prob: float = 0.0
                 ):
        """
            Args:
                in_chans: Number of channels in the input to the U-Net model.
                out_chans: Number of channels in the output to the U-Net model.
                chans: Number of output channels of the first convolution layer.
                num_pool_layers: Number of down-sampling and up-sampling layers.
                drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_channels = in_channels
        self.channels = channels

        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([
            ConvBlock(in_channels, channels, drop_prob)
        ])
        ch = channels

        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch*2, drop_prob))
            ch *= 2

        self.conv = ConvBlock(ch, ch*2, drop_prob)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
          Image: Input 4D tensor of shape (Batch Size, in channels, H, W)
        Returns:
          Output tensor of shape (Batch Size, out_channels, H, W)
        """
        output = image

        # Appplying down sample layers
        for num, layer in enumerate(self.down_sample_layers):
            output = layer(output)
            output = F.max_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)
        return output


# RoI Align, it was a mistake, I assumed RoIPool for RoIALign, but it was not the case

class RoIAlign(nn.Module):
    def __init__(self, output_size=(3, 3), spatial_scale=0.125, sampling_ratio=2):
        super().__init__()

        """
        Args
        output_size: (h, w) of the output feature map
        spatial_scale: ratio of the input feature map height (or w) to the raw image height (or w).
                        Equals the reciprocal of total stride in convolutional layers
        sampling_ratio: number of inputs samples to take for each output sample
        """

        # self.output_size = output_size
        # self.spatial_scale = spatial_scale
        # self.sampling_ratio = sampling_ratio
        self.roi_align = RoIAlign(
            output_size, spatial_scale=spatial_scale, sampling_ratio=sampling_ratio)

    def forward(self, image_embedding, bboxes):
        """
        Args:
          image_embedding: Input 4D tensor of shape (Batch size, in channels, H, W)
          bboxes: Input 3D Tensor of shape (Batch Size, max sequence length, 4) (4 corresponding to xmin, ymin, xmax, ymax)
        Returns:
          feature_maps_bboxes: tensor of shape (batch, max sequence length, in channels, *output_size)
        """

        feature_maps_bboxes = []
        for single_batch_img, single_batch_bbox in zip(image_embedding, bboxes):
            feature_map_single_batch = self.roi_align(input=single_batch_img.unsqueeze(0),
                                                      rois=torch.cat([torch.zeros(single_batch_bbox.shape[0], 1).to(
                                                          single_batch_bbox.device), single_batch_bbox], axis=-1).float()
                                                      )
            feature_maps_bboxes.append(feature_map_single_batch)

        return torch.stack(feature_maps_bboxes, axis=0)


# RoIPool

class RoIPool(nn.Module):

    def __init__(self, output_size=(3, 3), spatial_scale=0.125):
        super().__init__()
        """Args
        output_size: (h, w) of the output feature map
        spatial_scale: ratio of the input feature map height (or w) to the raw image height (or w).
                        Equals the reciprocal of total stride in convolutional layers
        """

        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.roi_pool = roi_pool

    def forward(self, image_embedding, bboxes):
        """
        Args:
          image_embedding: Input 4D tensor of shape (Batch size, in channels, H, W)
          bboxes: Input 3D Tensor of shape (Batch Size, max sequence length, 4) (4 corresponding to xmin, ymin, xmax, ymax)
        Returns:
          feature_maps_bboxes: tensor of shape (batch, max sequence length, in channels, *output_size)
        """

        feature_maps_bboxes = []
        for single_batch_img, single_batch_bbox in zip(image_embedding, bboxes):
            feature_map_single_batch = self.roi_pool(input=single_batch_img.unsqueeze(0),
                                                     boxes=torch.cat([torch.zeros(single_batch_bbox.shape[0], 1).to(
                                                         single_batch_bbox.device), single_batch_bbox], axis=-1).float(),
                                                     output_size=self.output_size,
                                                     spatial_scale=self.spatial_scale
                                                     )
            feature_maps_bboxes.append(feature_map_single_batch)

        return torch.stack(feature_maps_bboxes, axis=0)
