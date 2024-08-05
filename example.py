import numpy as np
import cv2
import torch
from torch import nn

import tapnet.utils as utils
from tapnet import tapir_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random = np.random.RandomState(2)


# class TapirPredictor(nn.Module):
#     def __init__(self, model, device):
#         super().__init__()
#         self.model = model
#         self.device = device
#
#     def forward(self, frame, query_points, causal_context, fast=True):
#         query_feats, hires_query_feats = online_model_init(frame, query_points)
#         return online_model_predict(frame, query_feats, hires_query_feats, causal_context, fast)
#
#     def build_model(self, model_path: str, input_resolution: tuple[int, int], pyramid_level: int = 1, use_casual_conv: bool = True):
#         model = tapir_model.TAPIR(pyramid_level=pyramid_level, use_casual_conv=use_casual_conv, initial_resolution=input_resolution, device=self.device)
#         model.load_state_dict(torch.load(model_path))
#         model = model.to(self.device)
#         model = model.eval()
#         return model


@torch.inference_mode()
def online_model_init(frame, query_points):
    """Initialize query features for the query points."""
    frame = utils.preprocess_frame(frame, resize=(resize_height, resize_width))
    feature_grid, hires_feats_grid = model.get_feature_grids(frame)

    feature_grid = feature_grid.cpu().numpy()
    hires_feats_grid = hires_feats_grid.cpu().numpy()

    query_feats, hires_query_feats = utils.get_query_features(
        query_points=query_points,
        feature_grid=feature_grid,
        hires_feats_grid=hires_feats_grid,
        initial_resolution=(resize_height, resize_width),
    )
    return query_feats, hires_query_feats


@torch.inference_mode()
def online_model_predict(frame, query_feats, hires_query_feats, causal_context):

    query_feats = torch.tensor(query_feats).to(device).float()
    hires_query_feats = torch.tensor(hires_query_feats).to(device).float()

    """Compute point tracks and occlusions given frame and query points."""
    frame = utils.preprocess_frame(frame, resize=(resize_height, resize_width))
    feature_grid, hires_feats_grid = model.get_feature_grids(frame)

    tracks, occlusions, expected_dist, causal_context = model.estimate_trajectories(
        feature_grid=feature_grid,
        hires_feats_grid=hires_feats_grid,
        query_feats=query_feats,
        hires_query_feats=hires_query_feats,
        causal_context=causal_context
    )
    visibles = utils.postprocess_occlusions(occlusions, expected_dist)
    return tracks, visibles, causal_context

if __name__ == '__main__':
    resize_height = 256
    resize_width = 256
    num_points = 256
    num_iters = 1

    model = tapir_model.TAPIR(use_casual_conv=True, num_pips_iter=num_iters, initial_resolution=(resize_height, resize_width), device=device)
    model.load_state_dict(torch.load('causal_bootstapir_checkpoint.pt'))
    model = model.to(device)
    model = model.eval()

    # cap = cv2.VideoCapture('https://storage.googleapis.com/dm-tapnet/horsejump-high.mp4')
    cap = cv2.VideoCapture('horsejump-high.mp4')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    query_points = utils.sample_grid_points(resize_height, resize_width, num_points)
    point_colors = random.randint(0, 255, (num_points, 3))

    # Initialize query features
    ret, frame = cap.read()
    query_feats, hires_query_feats = online_model_init(frame, query_points)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    causal_state = model.construct_initial_causal_state(num_iters, query_points.shape[0])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Note: we add a batch dimension.
        tracks, visibles, causal_state = online_model_predict(frame, query_feats, hires_query_feats, causal_state)
        visibles = visibles.cpu().numpy().squeeze()

        tracks = tracks.cpu().numpy().squeeze()
        tracks[:, 0] = tracks[:, 0] * width / resize_width
        tracks[:, 1] = tracks[:, 1] * height / resize_height

        frame = utils.draw_points(frame, tracks, visibles, point_colors)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

