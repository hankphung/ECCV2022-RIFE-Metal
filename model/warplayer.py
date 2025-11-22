import torch
import torch.nn as nn

# Support MPS (Metal Performance Shaders) for M1/M2 Macs
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
backwarp_tenGrid = {}


def warp(tenInput, tenFlow):
    # Use the device from tenFlow instead of global device
    flow_device = tenFlow.device
    k = (str(flow_device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=flow_device).view(
            1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=flow_device).view(
            1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat(
            [tenHorizontal, tenVertical], 1).to(flow_device)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    # MPS doesn't support 'border' padding, use 'zeros' instead
    padding_mode = 'zeros' if flow_device.type == 'mps' else 'border'
    return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode=padding_mode, align_corners=True)
