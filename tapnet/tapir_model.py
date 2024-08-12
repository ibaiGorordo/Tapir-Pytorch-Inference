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

"""TAPIR models definition."""

from typing import Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from tapnet import nets
from tapnet import utils


class TAPIR(nn.Module):
    """TAPIR model."""

    def __init__(
            self,
            bilinear_interp_with_depthwise_conv: bool = False,
            num_pips_iter: int = 4,
            pyramid_level: int = 1,
            num_mixer_blocks: int = 12,
            patch_size: int = 7,
            softmax_temperature: float = 20.0,
            parallelize_query_extraction: bool = False,
            initial_resolution: Tuple[int, int] = (256, 256),
            feature_extractor_chunk_size: int = 10,
            extra_convs: bool = True,
            use_casual_conv: bool = False,
            device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.highres_dim = 128
        self.lowres_dim = 256
        self.bilinear_interp_with_depthwise_conv = (
            bilinear_interp_with_depthwise_conv
        )
        self.parallelize_query_extraction = parallelize_query_extraction

        self.num_pips_iter = num_pips_iter
        self.pyramid_level = pyramid_level
        self.patch_size = patch_size
        self.softmax_temperature = softmax_temperature
        self.initial_resolution = tuple(initial_resolution)
        self.feature_extractor_chunk_size = feature_extractor_chunk_size
        self.num_mixer_blocks = num_mixer_blocks
        self.use_casual_conv = use_casual_conv
        self.device = device

        highres_dim = 128
        lowres_dim = 256
        strides = (1, 2, 2, 1)
        blocks_per_group = (2, 2, 2, 2)
        channels_per_group = (64, highres_dim, 256, lowres_dim)
        use_projection = (True, True, True, True)

        self.resnet_torch = nets.ResNet(
            blocks_per_group=blocks_per_group,
            channels_per_group=channels_per_group,
            use_projection=use_projection,
            strides=strides,
        )
        self.torch_cost_volume_track_mods = nn.ModuleDict({
            'hid1': torch.nn.Conv2d(1, 16, 3, 1, 1),
            'hid2': torch.nn.Conv2d(16, 1, 3, 1, 1),
            'hid3': torch.nn.Conv2d(16, 32, 3, 2, 0),
            'hid4': torch.nn.Linear(32, 16),
            'occ_out': torch.nn.Linear(16, 2),
        })
        dim = 4 + self.highres_dim + self.lowres_dim
        input_dim = dim + (self.pyramid_level + 2) * 49
        self.torch_pips_mixer = nets.PIPSMLPMixer(
            input_dim, dim, use_causal_conv=self.use_casual_conv
        )

        if extra_convs:
            self.extra_convs = nets.ExtraConvs()
        else:
            self.extra_convs = None

        self.avg_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)

    def get_feature_grids(self,frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        resnet_out = self.resnet_torch(frame)

        latent = resnet_out['resnet_unit_3'].permute(0, 2, 3, 1)
        hires = resnet_out['resnet_unit_1'].permute(0, 2, 3, 1)

        latent = self.extra_convs(latent)
        latent = latent / torch.sqrt(
            torch.maximum(
                torch.sum(torch.square(latent), dim=-1, keepdim=True),
                torch.full((), 1e-12, device=latent.device),
            )
        )
        hires = hires / torch.sqrt(
            torch.maximum(
                torch.sum(torch.square(hires), dim=-1, keepdim=True),
                torch.full((), 1e-12, device=hires.device),
            )
        )

        feature_grid = latent.view(1, *latent.shape[1:])
        hires_feats_grid = hires.view(1, *hires.shape[1:])

        return feature_grid, hires_feats_grid

    def estimate_trajectories(
            self,
            feature_grid: torch.Tensor,
            hires_feats_grid: torch.Tensor,
            query_feats: torch.Tensor,
            hires_query_feats: torch.Tensor,
            causal_context: torch.Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:

        num_iters = self.num_pips_iter

        num_queries = query_feats.shape[1]
        perm = torch.arange(num_queries)

        inv_perm = torch.zeros_like(perm)
        inv_perm[perm] = torch.arange(num_queries)

        points, occlusion, expected_dist = self.tracks_from_cost_volume(
            query_feats,
            feature_grid,
            None,
            im_shp=feature_grid.shape[0:1]
                   + self.initial_resolution
                   + (3,),
        )
        feature_grid_perm = feature_grid.permute(0, 3, 1, 2)
        new_causal_context = causal_context
        mixer_feats = None
        for i in range(num_iters):

            queries = [hires_query_feats, query_feats, query_feats]
            feature_grid_avg = self.avg_pool(feature_grid_perm)
            feature_grid_avg = feature_grid_avg.permute(0, 2, 3, 1)
            pyramid = [hires_feats_grid, feature_grid, feature_grid_avg]

            refined = self.refine_pips(
                queries,
                pyramid,
                points,
                occlusion,
                expected_dist,
                orig_hw=self.initial_resolution,
                last_iter=mixer_feats,
                causal_context=causal_context[i,...],
            )

            points, occlusion, expected_dist, mixer_feats, cc = refined
            new_causal_context[i,...] = cc

        return points, occlusion, expected_dist, causal_context

    def refine_pips(self, target_feature, pyramid,
                    pos_guess, occ_guess, expd_guess,
                    orig_hw, last_iter, causal_context) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:


        orig_h, orig_w = orig_hw
        corrs_pyr = []
        assert len(target_feature) == len(pyramid)
        for pyridx, (query, grid) in enumerate(zip(target_feature, pyramid)):
            # note: interp needs [y,x]
            coords = utils.convert_grid_coordinates(
                pos_guess, (orig_w, orig_h), grid.shape[-2:-4:-1]
            )
            coords = torch.flip(coords, dims=(-1,))
            last_iter_query = None
            if last_iter is not None:
                if pyridx == 0:
                    last_iter_query = last_iter[..., :self.highres_dim]
                else:
                    last_iter_query = last_iter[..., self.highres_dim:]

            ctxy, ctxx = torch.meshgrid(
                torch.arange(-3, 4), torch.arange(-3, 4), indexing='ij'
            )
            ctx = torch.stack([ctxy, ctxx], dim=-1)
            ctx = ctx.reshape(-1, 2).to(coords.device)
            coords2 = coords.unsqueeze(2) + ctx.unsqueeze(0).unsqueeze(0)
            neighborhood = utils.map_sampled_coordinates_2d(grid, coords2)

            # s is spatial context size
            if last_iter_query is None:
                patches = torch.einsum('bnsc,bnc->bns', neighborhood, query)
            else:
                patches = torch.einsum(
                    'bnsc,bnc->bns', neighborhood, last_iter_query
                )

            corrs_pyr.append(patches)
        corrs_pyr = torch.concatenate(corrs_pyr, dim=-1)

        corrs_chunked = corrs_pyr
        pos_guess_input = pos_guess
        occ_guess_input = occ_guess[..., None]
        expd_guess_input = expd_guess[..., None]

        # mlp_input is batch, num_points, num_chunks, frames_per_chunk, channels
        if last_iter is None:
            both_feature = torch.cat([target_feature[0], target_feature[1]], dim=-1)
            mlp_input_features = both_feature
        else:
            mlp_input_features = last_iter

        pos_guess_input = torch.zeros_like(pos_guess_input)
        mlp_input = torch.cat(
            [
                pos_guess_input,
                occ_guess_input,
                expd_guess_input,
                mlp_input_features,
                corrs_chunked,
            ],
            dim=-1,
        )
        b, n, c = mlp_input.shape
        x = mlp_input.view(b * n, c)

        res, new_causal_context = self.torch_pips_mixer(x.unsqueeze(1).float(), causal_context)

        n, _, c = res.shape
        b = mlp_input.shape[0]
        res = res.view(b, n, c)

        pos_update = res[..., :2]

        return (
            pos_update + pos_guess,
            res[..., 2] + occ_guess,
            res[..., 3] + expd_guess,
            res[..., 4:] + (mlp_input_features if last_iter is None else last_iter),
            new_causal_context.unsqueeze(0),
        )

    def tracks_from_cost_volume(
            self,
            interp_feature: torch.Tensor,
            feature_grid: torch.Tensor,
            query_points: Optional[torch.Tensor],
            im_shp=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Converts features into tracks by computing a cost volume.

        The computed cost volume will have shape
          [batch, num_queries, time, height, width], which can be very
          memory intensive.

        Args:
          interp_feature: A tensor of features for each query point, of shape
            [batch, num_queries, channels, heads].
          feature_grid: A tensor of features for the video, of shape [batch, time,
            height, width, channels, heads].
          query_points: When computing tracks, we assume these points are given as
            ground truth and we reproduce them exactly.  This is a set of points of
            shape [batch, num_points, 3], where each entry is [t, y, x] in frame/
            raster coordinates.
          im_shp: The shape of the original image, i.e., [batch, num_frames, time,
            height, width, 3].

        Returns:
          A 2-tuple of the inferred points (of shape
            [batch, num_points, num_frames, 2] where each point is [x, y]) and
            inferred occlusion (of shape [batch, num_points, num_frames], where
            each is a logit where higher means occluded)
        """

        mods = self.torch_cost_volume_track_mods
        cost_volume = torch.einsum(
            'bnc,bhwc->bnhw',
            interp_feature,
            feature_grid,
        )

        shape = cost_volume.shape
        batch_size, num_points = cost_volume.shape[1:3]
        b, n, h, w = cost_volume.shape
        cost_volume = cost_volume.view(b * n, h, w, 1)

        cost_volume = cost_volume.permute(0, 3, 1, 2)
        occlusion = mods['hid1'](cost_volume)
        occlusion = torch.nn.functional.relu(occlusion)

        pos = mods['hid2'](occlusion)
        pos = pos.permute(1, 0, 2, 3)

        pos_sm = pos.reshape(pos.size(0), pos.size(1), -1)

        softmaxed = F.softmax(pos_sm * self.softmax_temperature, dim=-1)
        pos = softmaxed.view_as(pos)

        points = utils.heatmaps_to_points(pos, im_shp, query_points=query_points)

        occlusion = torch.nn.functional.pad(occlusion, (0, 2, 0, 2))
        occlusion = mods['hid3'](occlusion)
        occlusion = torch.nn.functional.relu(occlusion)
        occlusion = torch.mean(occlusion, dim=(-1, -2))
        occlusion = mods['hid4'](occlusion)
        occlusion = torch.nn.functional.relu(occlusion)
        occlusion = mods['occ_out'](occlusion)

        n = shape[1]
        b = occlusion.shape[0] // n
        occlusion = occlusion.view(b, n, 2)
        expected_dist = occlusion[..., 1]
        occlusion = occlusion[..., 0]

        return points, occlusion, expected_dist