import numpy as np
import cv2
import torch

from tapnet.utils import sample_grid_points, preprocess_frame
from tapnet.tapir_inference import TapirPredictor, TapirPointEncoder, build_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random = np.random.RandomState(2)


def set_points(predictor: TapirPredictor, encoder: TapirPointEncoder, frame: torch.Tensor, query_points: torch.Tensor,
               device: torch.device):
    # Runs the model on the frame and randome query features, to get the feature grids
    # Then, it calculates the query features for the query points using the feature grids

    # Initialize causal state, query_feats and hires_query_feats
    causal_state_shape = (predictor.model.num_pips_iter, predictor.model.num_mixer_blocks, num_points, 2, 512 + 2048)
    causal_state = torch.zeros(causal_state_shape, dtype=torch.float32, device=device)
    query_feats = torch.zeros((1, num_points, 256), dtype=torch.float32, device=device)
    hires_query_feats = torch.zeros((1, num_points, 128), dtype=torch.float32, device=device)
    input_resolution = torch.tensor(frame.shape[2:]).to(device)

    _, _, _, feature_grid, hires_feats_grid = predictor(frame, query_feats, hires_query_feats, causal_state)

    query_feats, hires_query_feats = encoder(query_points[None], feature_grid, hires_feats_grid, input_resolution)

    return query_feats, hires_query_feats, causal_state


def draw_points(frame, points, visible, colors):
    for i in range(points.shape[0]):
        if not visible[i]:
            continue

        point = points[i, :]
        color = colors[i, :]
        cv2.circle(frame,
                   (int(point[0]), int(point[1])),
                   5,
                   (int(color[0]), int(color[1]), int(color[2])),
                   -1)
    return frame


if __name__ == '__main__':
    resize_height = 320
    resize_width = 320
    num_points = 625
    num_iters = 4

    query_points = sample_grid_points(resize_height, resize_width, num_points)
    query_points = torch.tensor(query_points).to(device)
    point_colors = random.randint(0, 255, (num_points, 3))

    cap = cv2.VideoCapture('https://storage.googleapis.com/dm-tapnet/horsejump-high.mp4')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model = build_model('causal_bootstapir_checkpoint.pt', (resize_height, resize_width), num_iters, True, device)
    predictor = TapirPredictor(model).to(device)
    enconder = TapirPointEncoder(model).to(device)

    # Initialize query features
    ret, frame = cap.read()
    input_frame = preprocess_frame(frame, resize=(resize_width, resize_height))
    query_feats, hires_query_feats, causal_state = set_points(predictor, enconder, input_frame, query_points, device)

    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

    # Reset video to the beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame
        input_frame = preprocess_frame(frame, resize=(resize_width, resize_height))

        # Run the model
        tracks, visibles, causal_state, _, _ = predictor(input_frame, query_feats, hires_query_feats, causal_state)

        # Postprocess frame
        visibles = visibles.cpu().numpy().squeeze()
        tracks = tracks.cpu().numpy().squeeze()

        tracks[:, 0] = tracks[:, 0] * width / resize_width
        tracks[:, 1] = tracks[:, 1] * height / resize_height

        frame = draw_points(frame, tracks, visibles, point_colors)
        out.write(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
