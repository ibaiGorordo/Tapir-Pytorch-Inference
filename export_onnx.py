import argparse
import onnx
import torch
from onnxsim import simplify

from tapnet.tapir_inference import TapirPredictor, TapirPointEncoder, build_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_parser():
    parser = argparse.ArgumentParser(description="Tapir ONNX Export")
    parser.add_argument("--model", default="models/causal_bootstapir_checkpoint.pt", type=str,
                        help="path to Tapir checkpoint")
    parser.add_argument("--resolution", default=640, type=int, help="Input resolution")
    parser.add_argument("--num_points", default=1000, type=int, help="Number of points")
    parser.add_argument("--num_iters", default=4, type=int, help="Number of iterations, 1 for faster inference, 4 for better results")
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    resolution = args.resolution
    num_points = args.num_points
    num_iters = args.num_iters

    model = build_model('models/causal_bootstapir_checkpoint.pt',
                        (resolution, resolution), num_iters, True, device).eval()
    predictor = TapirPredictor(model).to(device).eval()
    encoder = TapirPointEncoder(model).to(device).eval()

    causal_state_shape = (num_iters, model.num_mixer_blocks, num_points, 2, 512 + 2048)
    causal_state = torch.zeros(causal_state_shape, dtype=torch.float32, device=device)
    feature_grid = torch.zeros((1, resolution//8, resolution//8, 256), dtype=torch.float32, device=device)
    hires_feats_grid = torch.zeros((1, resolution//4, resolution//4, 128), dtype=torch.float32, device=device)
    query_points = torch.zeros((num_points, 2), dtype=torch.float32, device=device)
    input_resolution = torch.tensor((resolution, resolution)).to(device)
    input_frame = torch.zeros((1, 3, resolution, resolution), dtype=torch.float32, device=device)

    # Test model
    query_feats, hires_query_feats = encoder(query_points[None], feature_grid, hires_feats_grid, input_resolution)
    tracks, visibles, causal_state, _, _ = predictor(input_frame, query_feats, hires_query_feats, causal_state)

    # Export model
    torch.onnx.export(encoder,
                        (query_points[None], feature_grid, hires_feats_grid, input_resolution),
                        'tapir_pointencoder.onnx',
                        verbose=True,
                            input_names=['query_points', 'feature_grid', 'hires_feats_grid', 'input_resolution'],
                            output_names=['query_feats', 'hires_query_feats'])

    torch.onnx.export(predictor,
                      (input_frame, query_feats, hires_query_feats, causal_state),
                      'tapir.onnx',
                      verbose=True,
                        input_names=['input_frame', 'query_feats', 'hires_query_feats', 'causal_state'],
                        output_names=['tracks', 'visibles', 'causal_state', 'feature_grid', 'hires_feats_grid'])

    # Simplify model
    model_simp, check = simplify('tapir_pointencoder.onnx')
    onnx.save(model_simp, 'tapir_pointencoder.onnx')

    model_simp, check = simplify('tapir.onnx')
    onnx.save(model_simp, 'tapir.onnx')



