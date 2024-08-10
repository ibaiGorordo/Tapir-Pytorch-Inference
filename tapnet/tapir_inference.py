import numpy as np
import torch
from torch import nn
from tapnet.tapir_model import TAPIR
from tapnet.utils import get_query_features, postprocess_occlusions, preprocess_frame


def build_model(model_path: str, input_resolution: tuple[int, int], num_pips_iter: int, use_casual_conv: bool,
                device: torch.device):
    model = TAPIR(use_casual_conv=use_casual_conv, num_pips_iter=num_pips_iter,
                  initial_resolution=input_resolution, device=device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to(device)
    model = model.eval()
    return model


class TapirPredictor(nn.Module):
    def __init__(self, model: TAPIR):
        super().__init__()
        self.model = model

    @torch.inference_mode()
    def forward(self, frame, query_feats, hires_query_feats, causal_context):
        feature_grid, hires_feats_grid = self.model.get_feature_grids(frame)

        tracks, occlusions, expected_dist, causal_context = self.model.estimate_trajectories(
            feature_grid=feature_grid,
            hires_feats_grid=hires_feats_grid,
            query_feats=query_feats,
            hires_query_feats=hires_query_feats,
            causal_context=causal_context
        )
        visibles = postprocess_occlusions(occlusions, expected_dist)
        return tracks, visibles, causal_context, feature_grid, hires_feats_grid


class TapirPointEncoder(nn.Module):
    def __init__(self, model: TAPIR):
        super().__init__()
        self.model = model

    @torch.inference_mode()
    def forward(self, query_points, feature_grid, hires_feats_grid):
        return get_query_features(query_points, feature_grid, hires_feats_grid)


class TapirInference(nn.Module):
    def __init__(self, model_path: str, input_resolution: tuple[int, int], num_pips_iter: int, device: torch.device):
        super().__init__()
        self.model = build_model(model_path, input_resolution, num_pips_iter, True, device)
        self.predictor = TapirPredictor(self.model).to(device)
        self.encoder = TapirPointEncoder(self.model).to(device)
        self.device = device

        self.num_points = 256
        causal_state_shape = (num_pips_iter, self.model.num_mixer_blocks, self.num_points, 2, 512 + 2048)
        self.causal_state = torch.zeros(causal_state_shape, dtype=torch.float32, device=self.device)
        self.query_feats = torch.zeros((1, self.num_points, 256), dtype=torch.float32, device=self.device)
        self.hires_query_feats = torch.zeros((1, self.num_points, 128), dtype=torch.float32, device=self.device)

    def set_points(self, frame: np.ndarray, query_points: np.ndarray):
        query_points = query_points.astype(np.float32)
        query_points[..., 0] = query_points[..., 0] / self.input_resolution[1]
        query_points[..., 1] = query_points[..., 1] / self.input_resolution[0]

        query_points = torch.tensor(query_points).to(self.device)

        num_points = query_points.shape[0]
        causal_state_shape = (self.model.num_pips_iter,  self.model.num_mixer_blocks, num_points, 2, 512 + 2048)
        self.causal_state = torch.zeros(causal_state_shape, dtype=torch.float32, device=self.device)
        query_feats = torch.zeros((1, num_points, 256), dtype=torch.float32, device=self.device)
        hires_query_feats = torch.zeros((1, num_points, 128), dtype=torch.float32, device=self.device)

        input_frame = preprocess_frame(frame, resize=self.input_resolution, device=self.device)
        _, _, _, feature_grid, hires_feats_grid = self.predictor(input_frame, query_feats, hires_query_feats, self.causal_state)

        self.query_feats, self.hires_query_feats = self.encoder(query_points[None], feature_grid, hires_feats_grid)

    @torch.inference_mode()
    def forward(self, frame: np.ndarray):
        height, width = frame.shape[:2]

        input_frame = preprocess_frame(frame, resize=self.input_resolution, device=self.device)
        tracks, visibles, self.causal_state, _, _ = self.predictor(input_frame, self.query_feats,
                                                                   self.hires_query_feats, self.causal_state)

        visibles = visibles.cpu().numpy().squeeze()
        tracks = tracks.cpu().numpy().squeeze()

        tracks[:, 0] = tracks[:, 0] * width / self.input_resolution[1]
        tracks[:, 1] = tracks[:, 1] * height / self.input_resolution[0]

        return tracks, visibles
