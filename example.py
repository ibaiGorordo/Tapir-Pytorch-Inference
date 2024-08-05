import numpy as np
import cv2
import torch
from torch import nn

import tapnet.utils as utils
from tapnet import tapir_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random = np.random.RandomState(2)


def build_model(model_path: str, input_resolution: tuple[int, int], num_pips_iter: int, use_casual_conv: bool,
                device: torch.device):
    model = tapir_model.TAPIR(use_casual_conv=use_casual_conv, num_pips_iter=num_pips_iter,
                              initial_resolution=input_resolution, device=device)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model = model.eval()
    return model


def set_points(predictor, frame, query_points):
    # Runs the model on the frame and randome query features, to get the feature grids
    # Then, it calculates the query features for the query points using the feature grids

    # Initialize causal state, feature grid and hires feature grid
    causal_state = torch.zeros(
        (predictor.model.num_pips_iter, predictor.model.num_mixer_blocks, num_points, 2, 512 + 2048),
        dtype=torch.float32, device=predictor.device)
    query_feats = torch.zeros((1, 256, 256), dtype=torch.float32, device=predictor.device)
    hires_query_feats = torch.zeros((1, 256, 128), dtype=torch.float32, device=predictor.device)

    _, _, _, feature_grid, hires_feats_grid = predictor(frame, query_feats, hires_query_feats, causal_state)

    feature_grid = feature_grid.cpu().numpy()
    hires_feats_grid = hires_feats_grid.cpu().numpy()

    query_feats, hires_query_feats = utils.get_query_features(query_points, feature_grid, hires_feats_grid,
                                                              predictor.input_resolution)

    query_feats = torch.tensor(query_feats).to(predictor.device).float()
    hires_query_feats = torch.tensor(hires_query_feats).to(predictor.device).float()

    return query_feats, hires_query_feats, causal_state


class TapirPredictor(nn.Module):
    def __init__(self, model_path: str, input_resolution: tuple[int, int], num_pips_iter: int = 4,
                 use_casual_conv: bool = True, device: torch.device = device):
        super().__init__()
        self.model = build_model(model_path, input_resolution, num_pips_iter, use_casual_conv, device)
        self.device = device
        self.input_resolution = input_resolution

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
        visibles = utils.postprocess_occlusions(occlusions, expected_dist)
        return tracks, visibles, causal_context, feature_grid, hires_feats_grid


if __name__ == '__main__':
    resize_height = 256
    resize_width = 256
    num_points = 256
    num_iters = 4

    query_points = utils.sample_grid_points(resize_height, resize_width, num_points)
    point_colors = random.randint(0, 255, (num_points, 3))

    # cap = cv2.VideoCapture('https://storage.googleapis.com/dm-tapnet/horsejump-high.mp4')
    cap = cv2.VideoCapture('horsejump-high.mp4')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    predictor = TapirPredictor('causal_bootstapir_checkpoint.pt', (resize_height, resize_width),
                               num_pips_iter=num_iters, device=device)

    # Initialize query features
    ret, frame = cap.read()
    input_frame = utils.preprocess_frame(frame)
    query_feats, hires_query_feats, causal_state = set_points(predictor, input_frame, query_points)

    # Reset video to the beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        """Compute point tracks and occlusions given frame and query points."""
        input_frame = utils.preprocess_frame(frame, resize=(resize_height, resize_width))

        # Note: we add a batch dimension.
        tracks, visibles, causal_state, _, _ = predictor(input_frame, query_feats, hires_query_feats, causal_state)

        visibles = visibles.cpu().numpy().squeeze()
        tracks = tracks.cpu().numpy().squeeze()
        tracks[:, 0] = tracks[:, 0] * width / resize_width
        tracks[:, 1] = tracks[:, 1] * height / resize_height

        frame = utils.draw_points(frame, tracks, visibles, point_colors)

        cv2.imshow('frame', frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
