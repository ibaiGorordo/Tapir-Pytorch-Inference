# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Pytorch model utilities."""

from typing import Any, Sequence, Union
import numpy as np
import cv2
import torch
import torch.nn.functional as F

random = np.random.RandomState(2)

def bilinear(x: torch.Tensor, resolution: tuple[int, int]) -> torch.Tensor:
    """Resizes a 5D tensor using bilinear interpolation.

    Args:
          x: A 5D tensor of shape (B, T, W, H, C) where B is batch size, T is
            time, W is width, H is height, and C is the number of channels.
      resolution: The target resolution as a tuple (new_width, new_height).

    Returns:
      The resized tensor.
    """
    b, t, h, w, c = x.size()
    x = x.permute(0, 1, 4, 2, 3).reshape(b, t * c, h, w)
    x = F.interpolate(x, size=resolution, mode='bilinear', align_corners=False)
    b, _, h, w = x.size()
    x = x.reshape(b, t, c, h, w).permute(0, 1, 3, 4, 2)
    return x


def map_coordinates_3d(
        feats: torch.Tensor, coordinates: torch.Tensor
) -> torch.Tensor:
    """Maps 3D coordinates to corresponding features using bilinear interpolation.

    Args:
      feats: A 5D tensor of features with shape (B, W, H, D, C), where B is batch
        size, W is width, H is height, D is depth, and C is the number of
        channels.
      coordinates: A 3D tensor of coordinates with shape (B, N, 3), where N is the
        number of coordinates and the last dimension represents (W, H, D)
        coordinates.

    Returns:
      The mapped features tensor.
    """
    x = feats.permute(0, 4, 1, 2, 3)
    y = coordinates[:, :, None, None, :].float()
    y[..., 0] += 0.5
    y = 2 * (y / torch.tensor(x.shape[2:], device=y.device)) - 1
    y = torch.flip(y, dims=(-1,))
    out = (
        F.grid_sample(
            x, y, mode='bilinear', align_corners=False, padding_mode='border'
        )
    )
    out = out.squeeze(dim=(3, 4))
    out = out.permute(0, 2, 1)
    return out


def map_coordinates_2d(feats: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    n, h, w, c = feats.shape
    x = feats.permute(0, 3, 1, 2)

    y = coordinates[:, :, None, :]
    y = 2 * (y / torch.tensor(x.shape[2:], device=y.device)) - 1
    y = torch.flip(y, dims=(-1,)).float()

    out = F.grid_sample(
        x, y, mode='bilinear', align_corners=False, padding_mode='border'
    )
    out = out.squeeze(dim=-1)
    out = out.permute(0, 2, 1)
    return out


def map_smpled_coordinates_2d(feats: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    n, h, w, c = feats.shape
    x = feats.permute(0, 3, 1, 2).view(n, c, h, w)

    y = coordinates

    y = 2 * (y / h) - 1
    y = torch.flip(y, dims=(-1,)).float()

    out = F.grid_sample(
        x, y, mode='bilinear', align_corners=False, padding_mode='zeros'
    )

    out = out.permute(0, 2, 3, 1)
    return out


def soft_argmax_heatmap_batched(softmax_val, threshold=5):
    """Test if two image resolutions are the same."""
    b, h, d1, d2 = softmax_val.shape
    y, x = torch.meshgrid(
        torch.arange(d1, device=softmax_val.device),
        torch.arange(d2, device=softmax_val.device),
        indexing='ij',
    )
    coords = torch.stack([x + 0.5, y + 0.5], dim=-1).to(softmax_val.device)
    softmax_val_flat = softmax_val.reshape(b, h, -1)
    argmax_pos = torch.argmax(softmax_val_flat, dim=-1)

    pos = coords.reshape(-1, 2)[argmax_pos]
    valid = (
            torch.sum(
                torch.square(
                    coords[None, None, :, :, :] - pos[:, :, None, None, :]
                ),
                dim=-1,
                keepdims=True,
            )
            < threshold ** 2
    )

    weighted_sum = torch.sum(
        coords[None, None, :, :, :]
        * valid
        * softmax_val[:, :, :, :, None],
        dim=(2, 3),
    )

    sum_of_weights = torch.maximum(
        torch.sum(valid * softmax_val[:, :, :, :, None], dim=(2, 3)),
        torch.tensor(1e-12, device=softmax_val.device),
    )
    return weighted_sum / sum_of_weights


def heatmaps_to_points(
        all_pairs_softmax,
        image_shape,
        threshold=5,
        query_points=None,
):
    """Convert heatmaps to points using soft argmax."""

    out_points = soft_argmax_heatmap_batched(all_pairs_softmax, threshold)
    feature_grid_shape = all_pairs_softmax.shape[1:]


    # Note: out_points is now [x, y]; we need to divide by [width, height].
    # image_shape[3] is width and image_shape[2] is height.
    out_points = convert_grid_coordinates(
        out_points.detach(),
        feature_grid_shape[1:3],
        image_shape[1:3],
    )
    return out_points


def is_same_res(r1, r2):
    """Test if two image resolutions are the same."""
    return all([x == y for x, y in zip(r1, r2)])


def convert_grid_coordinates(
        coords: torch.Tensor,
        input_grid_size: Sequence[int],
        output_grid_size: Sequence[int],
        coordinate_format: str = 'xy',
) -> torch.Tensor:
    """Convert grid coordinates to correct format."""
    if isinstance(input_grid_size, tuple):
        input_grid_size = torch.tensor(input_grid_size, device=coords.device)
    if isinstance(output_grid_size, tuple):
        output_grid_size = torch.tensor(output_grid_size, device=coords.device)

    if coordinate_format == 'xy':
        if input_grid_size.shape[0] != 2 or output_grid_size.shape[0] != 2:
            raise ValueError(
                'If coordinate_format is xy, the shapes must be length 2.'
            )
    elif coordinate_format == 'tyx':
        if input_grid_size.shape[0] != 3 or output_grid_size.shape[0] != 3:
            raise ValueError(
                'If coordinate_format is tyx, the shapes must be length 3.'
            )
        if input_grid_size[0] != output_grid_size[0]:
            raise ValueError('converting frame count is not supported.')
    else:
        raise ValueError('Recognized coordinate formats are xy and tyx.')

    position_in_grid = coords
    position_in_grid = position_in_grid * output_grid_size / input_grid_size

    return position_in_grid


def generate_default_resolutions(full_size, train_size, num_levels=None):
    """Generate a list of logarithmically-spaced resolutions.

    Generated resolutions are between train_size and full_size, inclusive, with
    num_levels different resolutions total.  Useful for generating the input to
    refinement_resolutions in PIPs.

    Args:
      full_size: 2-tuple of ints.  The full image size desired.
      train_size: 2-tuple of ints.  The smallest refinement level.  Should
        typically match the training resolution, which is (256, 256) for TAPIR.
      num_levels: number of levels.  Typically each resolution should be less than
        twice the size of prior resolutions.

    Returns:
      A list of resolutions.
    """
    if all([x == y for x, y in zip(train_size, full_size)]):
        return [train_size]

    if num_levels is None:
        size_ratio = np.array(full_size) / np.array(train_size)
        num_levels = int(np.ceil(np.max(np.log2(size_ratio))) + 1)

    if num_levels <= 1:
        return [train_size]

    h, w = full_size[0:2]
    if h % 8 != 0 or w % 8 != 0:
        print(
            'Warning: output size is not a multiple of 8. Final layer '
            + 'will round size down.'
        )
    ll_h, ll_w = train_size[0:2]

    sizes = []
    for i in range(num_levels):
        size = (
            int(round((ll_h * (h / ll_h) ** (i / (num_levels - 1))) // 8)) * 8,
            int(round((ll_w * (w / ll_w) ** (i / (num_levels - 1))) // 8)) * 8,
        )
        sizes.append(size)
    return sizes


def draw_points(frame, points, visible, colors):
    for i in range(points.shape[0]):
        if not visible[i]:
            continue

        point = points[i,:]
        color = colors[i,:]
        cv2.circle(frame,
                   (int(point[0]), int(point[1])),
                   5,
                   (int(color[0]), int(color[1]), int(color[2])),
                   -1)
    return frame

def preprocess_frame(frame, resize=(256, 256), device='cuda'):

    input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input = cv2.resize(input, resize)
    input = input[np.newaxis, :, :, :].astype(np.float32)

    input = torch.tensor(input).to(device)
    input = input.float()
    input = input / 255 * 2 - 1
    input = input.permute(0, 3, 1, 2)

    return input

def sample_grid_points(height, width, num_points):
    """Sample random points with (time, height, width) order."""
    x = np.linspace(0, width - 1, int(np.sqrt(num_points)))
    y = np.linspace(0, height - 1, int(np.sqrt(num_points)))
    x, y = np.meshgrid(x, y)
    x = np.expand_dims(x.flatten(), -1)
    y = np.expand_dims(y.flatten(), -1)
    points = np.concatenate((y, x), axis=-1).astype(np.int32)  # [num_points, 2]
    return points

def sample_random_points(height, width, num_points):
    x = random.randint(0, width-1, int(np.sqrt(num_points)))
    y = random.randint(0, height-1, int(np.sqrt(num_points)))
    x, y = np.meshgrid(x, y)
    x = np.expand_dims(x.flatten(), -1)
    y = np.expand_dims(y.flatten(), -1)
    points = np.concatenate((y, x), axis=-1).astype(np.int32)  # [num_points, 2]
    return points

def postprocess_occlusions(occlusions, expected_dist):
    visibles = (1 - F.sigmoid(occlusions)) * (1 - F.sigmoid(expected_dist)) > 0.5
    return visibles

def get_query_features(query_points: torch.Tensor,
                       feature_grid: torch.Tensor,
                       hires_feats_grid: torch.Tensor,
                       initial_resolution: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:

    position_in_grid = convert_grid_coordinates(
        query_points,
        torch.tensor(initial_resolution).to(query_points.device),
        feature_grid.shape[1:3],
        coordinate_format='xy',
    )

    position_in_grid_hires = convert_grid_coordinates(
        query_points,
        torch.tensor(initial_resolution).to(query_points.device),
        hires_feats_grid.shape[1:3],
        coordinate_format='xy',
    )
    query_feats = map_coordinates_2d(
        feature_grid, position_in_grid
    )
    hires_query_feats = map_coordinates_2d(
        hires_feats_grid, position_in_grid_hires
    )
    return query_feats, hires_query_feats
