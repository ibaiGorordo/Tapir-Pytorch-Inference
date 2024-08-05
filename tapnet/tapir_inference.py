import torch
from torch import nn
from .tapir_model import TAPIR
from .utils import get_query_features, postprocess_occlusions


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
    def forward(self, query_points, feature_grid, hires_feats_grid, initial_resolution):
        return get_query_features(query_points, feature_grid, hires_feats_grid, initial_resolution)


