import numpy as np
import cv2
import torch

from tapnet.utils import sample_grid_points, preprocess_frame
from tapnet.tapir_inference import TapirPredictor, TapirPointEncoder, build_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random = np.random.RandomState(2)


if __name__ == '__main__':
    resize_height = 256
    resize_width = 256
    num_points = 256
    num_iters = 4

    model = build_model('causal_bootstapir_checkpoint.pt', (resize_height, resize_width), num_iters, True, device)
    predictor = TapirPredictor(model).to(device)
    encoder = TapirPointEncoder(model).to(device)

    causal_state_shape = (num_iters, model.num_mixer_blocks, num_points, 2, 512 + 2048)
    causal_state = torch.zeros(causal_state_shape, dtype=torch.float32, device=device)
    feature_grid = torch.zeros((1, 32, 32, 256), dtype=torch.float32, device=device)
    hires_feats_grid = torch.zeros((1, 64, 64, 128), dtype=torch.float32, device=device)
    query_points = torch.zeros((num_points, 2), dtype=torch.float32, device=device)
    input_resolution = torch.tensor((resize_height, resize_width)).to(device)
    input_frame = torch.zeros((1, 3, resize_height, resize_width), dtype=torch.float32, device=device)


    # Test model
    query_feats, hires_query_feats = encoder(query_points[None], feature_grid, hires_feats_grid, input_resolution)
    tracks, visibles, causal_state, _, _ = predictor(input_frame, query_feats, hires_query_feats, causal_state)

    print(query_points[None].shape, feature_grid.shape, hires_feats_grid.shape, input_resolution.shape)
    print(query_feats.shape, hires_query_feats.shape)
    print(tracks.shape, visibles.shape, causal_state.shape, feature_grid.shape, hires_feats_grid.shape)

    # Export model
    torch.onnx.export(encoder,
                        (query_points[None], feature_grid, hires_feats_grid, input_resolution),
                        'tapir_encoder.onnx',
                        verbose=True,
                            input_names=['query_points', 'feature_grid', 'hires_feats_grid', 'input_resolution'],
                            output_names=['query_feats', 'hires_query_feats'])

    torch.onnx.export(predictor,
                      (input_frame, query_feats, hires_query_feats, causal_state),
                      'tapir.onnx',
                      verbose=True,
                        input_names=['input_frame', 'query_feats', 'hires_query_feats', 'causal_state'],
                        output_names=['tracks', 'visibles', 'causal_state', 'feature_grid', 'hires_feats_grid'])



