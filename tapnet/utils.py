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

def map_coordinates_2d(feats: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
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

def map_sampled_coordinates_2d(feats: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    n, h, w, c = feats.shape
    x = feats.permute(0, 3, 1, 2)

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
                keepdim=True,
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

def convert_grid_coordinates(
        coords: torch.Tensor,
        input_grid_size: Sequence[int],
        output_grid_size: Sequence[int]
) -> torch.Tensor:
    """Convert grid coordinates to correct format."""
    if isinstance(input_grid_size, tuple):
        input_grid_size = torch.tensor(input_grid_size, device=coords.device)
    if isinstance(output_grid_size, tuple):
        output_grid_size = torch.tensor(output_grid_size, device=coords.device)

    position_in_grid = coords
    position_in_grid = position_in_grid * output_grid_size / input_grid_size

    return position_in_grid


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
    x = random.randint(0, width-1, (num_points, 1))
    y = random.randint(0, height-1, (num_points, 1))
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
        initial_resolution,
        feature_grid.shape[1:3]
    )

    position_in_grid_hires = convert_grid_coordinates(
        query_points,
        initial_resolution,
        hires_feats_grid.shape[1:3]
    )

    query_feats = map_coordinates_2d(
        feature_grid, position_in_grid
    )
    hires_query_feats = map_coordinates_2d(
        hires_feats_grid, position_in_grid_hires
    )
    return query_feats, hires_query_feats

def draw_points(frame, points, visible, colors):
    for i in range(points.shape[0]):
        if not visible[i]:
            continue

        point = points[i, :]
        color = colors[i, :].astype(np.uint8).tolist()
        cv2.circle(frame, (int(point[0]), int(point[1])), 3, color, -1)
    return frame