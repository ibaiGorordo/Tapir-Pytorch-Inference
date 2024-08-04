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

from typing import Any, Mapping, Optional, Tuple

import torch
from torch import nn
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

    def get_query_features(
            self,
            query_points: torch.Tensor,
            feature_grid: torch.Tensor,
            hires_feats_grid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        position_in_grid = utils.convert_grid_coordinates(
            query_points,
            torch.tensor([self.initial_resolution[0], self.initial_resolution[1]]).to(query_points.device),
            feature_grid.shape[1:3],
            coordinate_format='xy',
        )

        position_in_grid_hires = utils.convert_grid_coordinates(
            query_points,
            torch.tensor([self.initial_resolution[0], self.initial_resolution[1]]).to(query_points.device),
            hires_feats_grid.shape[1:3],
            coordinate_format='xy',
        )
        query_feats = utils.map_coordinates_2d(
            feature_grid, position_in_grid
        )
        hires_query_feats = utils.map_coordinates_2d(
            hires_feats_grid, position_in_grid_hires
        )
        return query_feats, hires_query_feats

    def get_feature_grids(self,frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        resnet_out = self.resnet_torch(frame)

        latent = resnet_out['resnet_unit_3'].permute(0, 2, 3, 1)
        hires = resnet_out['resnet_unit_1'].permute(0, 2, 3, 1)

        latent = self.extra_convs(latent)
        latent = latent / torch.sqrt(
            torch.maximum(
                torch.sum(torch.square(latent), dim=-1, keepdim=True),
                torch.tensor(1e-12, device=latent.device),
            )
        )
        hires = hires / torch.sqrt(
            torch.maximum(
                torch.sum(torch.square(hires), dim=-1, keepdim=True),
                torch.tensor(1e-12, device=hires.device),
            )
        )

        feature_grid = latent.view(1, *latent.shape[1:])
        hires_feats_grid = hires.view(1, *hires.shape[1:])

        return feature_grid, hires_feats_grid

    def estimate_trajectories(
            self,
            video_size: Tuple[int, int],
            feature_grid: torch.Tensor,
            hires_feats_grid: torch.Tensor,
            query_feats: torch.Tensor,
            hires_query_feats: torch.Tensor,
            query_chunk_size: Optional[int] = None,
            causal_context: Optional[dict[str, torch.Tensor]] = None,
            get_causal_context: bool = False,
    ) -> Mapping[str, Any]:

        num_iters = self.num_pips_iter
        occ_iters = [[] for _ in range(num_iters + 1)]
        pts_iters = [[] for _ in range(num_iters + 1)]
        expd_iters = [[] for _ in range(num_iters + 1)]
        new_causal_context = [[] for _ in range(num_iters)]

        num_queries = query_feats.shape[1]
        perm = torch.arange(num_queries)

        inv_perm = torch.zeros_like(perm)
        inv_perm[perm] = torch.arange(num_queries)

        for ch in range(0, num_queries, query_chunk_size):
            perm_chunk = perm[ch: ch + query_chunk_size]
            chunk = query_feats[:, perm_chunk]

            cc_chunk = []
            for d in range(len(causal_context)):
                tmp_dict = {}
                for k, v in causal_context[d].items():
                    tmp_dict[k] = v[:, perm_chunk]
                cc_chunk.append(tmp_dict)

            points, occlusion, expected_dist = self.tracks_from_cost_volume(
                chunk,
                feature_grid,
                None,
                im_shp=feature_grid.shape[0:1]
                       + self.initial_resolution
                       + (3,),
            )

            coords = utils.convert_grid_coordinates(
                points,
                self.initial_resolution[::-1],
                video_size[::-1],
                coordinate_format='xy')

            pts_iters[0].append(coords)
            occ_iters[0].append(occlusion)
            expd_iters[0].append(expected_dist)

            mixer_feats = None
            for i in range(num_iters):
                queries = [
                    hires_query_feats[:, perm_chunk],
                    query_feats[:, perm_chunk],
                ]

                for _ in range(self.pyramid_level):
                    queries.append(queries[-1])
                pyramid = [
                    hires_feats_grid,
                    feature_grid,
                ]
                for _ in range(self.pyramid_level):
                    pyramid.append(
                        F.avg_pool3d(
                            pyramid[-1],
                            kernel_size=(2, 2, 1),
                            stride=(2, 2, 1),
                            padding=0,
                        )
                    )
                cc = cc_chunk[i]

                refined = self.refine_pips(
                    queries,
                    pyramid,
                    points,
                    occlusion,
                    expected_dist,
                    orig_hw=self.initial_resolution,
                    last_iter=mixer_feats,
                    resize_hw=self.initial_resolution,
                    causal_context=cc,
                    get_causal_context=get_causal_context,
                )

                points, occlusion, expected_dist, mixer_feats, cc = refined
                coords = utils.convert_grid_coordinates(
                    points,
                    self.initial_resolution[::-1],
                    video_size[::-1],
                    coordinate_format='xy')
                pts_iters[i + 1].append(coords)
                occ_iters[i + 1].append(occlusion)
                expd_iters[i + 1].append(expected_dist)
                new_causal_context[i].append(cc)

                if (i + 1) % self.num_pips_iter == 0:
                    mixer_feats = None
                    expected_dist = expd_iters[0][-1]
                    occlusion = occ_iters[0][-1]

        occlusion = []
        points = []
        expd = []
        for i, _ in enumerate(occ_iters):
            occlusion.append(torch.cat(occ_iters[i], dim=1)[:, inv_perm])
            points.append(torch.cat(pts_iters[i], dim=1)[:, inv_perm])
            expd.append(torch.cat(expd_iters[i], dim=1)[:, inv_perm])

        for i in range(len(new_causal_context)):
            combined_dict = {}
            for key in new_causal_context[i][0].keys():
                arrays = [d[key] for d in new_causal_context[i]]
                concatenated = torch.cat(arrays, dim=1)
                combined_dict[key] = concatenated[:, inv_perm]
            new_causal_context[i] = combined_dict

        out = dict(
            occlusion=occlusion,
            tracks=points,
            expected_dist=expd,
        )

        out['causal_context'] = new_causal_context
        return out

    def refine_pips(self,
            target_feature,
            pyramid,
            pos_guess,
            occ_guess,
            expd_guess,
            orig_hw,
            last_iter=None,
            resize_hw=None,
            causal_context=None,
            get_causal_context=False):

        orig_h, orig_w = orig_hw
        resized_h, resized_w = resize_hw
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
                    last_iter_query = last_iter[..., : self.highres_dim]
                else:
                    last_iter_query = last_iter[..., self.highres_dim:]
                #
                # print(last_iter_query.shape)

            ctxy, ctxx = torch.meshgrid(
                torch.arange(-3, 4), torch.arange(-3, 4), indexing='ij'
            )
            ctx = torch.stack([ctxy, ctxx], dim=-1)
            ctx = ctx.reshape(-1, 2).to(coords.device)
            # print(pyridx, coords.shape, ctx.shape)
            coords2 = coords.unsqueeze(2) + ctx.unsqueeze(0).unsqueeze(0)
            neighborhood = utils.map_smpled_coordinates_2d(grid, coords2)

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

        for k, v in causal_context.items():
            b, n, *_ = v.shape
            causal_context[k] = v.view(b * n, *v.shape[2:])

        res, new_causal_context = self.torch_pips_mixer(
            x.unsqueeze(1).float(), causal_context, get_causal_context
        )

        n, _, c = res.shape
        b = mlp_input.shape[0]
        res = res.view(b, n, c)

        for k, v in new_causal_context.items():
            b = mlp_input.shape[0]
            n = v.shape[0] // b
            new_causal_context[k] = v.view(b, n, *v.shape[1:])

        pos_update = utils.convert_grid_coordinates(
            res[..., :2],
            (resized_w, resized_h),
            (orig_w, orig_h),
        )
        return (
            pos_update + pos_guess,
            res[..., 2] + occ_guess,
            res[..., 3] + expd_guess,
            res[..., 4:] + (mlp_input_features if last_iter is None else last_iter),
            new_causal_context,
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

    def construct_initial_causal_state(self, num_points, num_resolutions=1):
        """Construct initial causal state."""
        value_shapes = {}
        for i in range(self.num_mixer_blocks):
            value_shapes[f'block_{i}_causal_1'] = (1, num_points, 2, 512)
            value_shapes[f'block_{i}_causal_2'] = (1, num_points, 2, 2048)
        fake_ret = {
            k: torch.zeros(v, dtype=torch.float32) for k, v in value_shapes.items()
        }
        return [fake_ret] * num_resolutions * 4
