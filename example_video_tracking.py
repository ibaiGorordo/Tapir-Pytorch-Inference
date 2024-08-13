import cv2
import torch
import numpy as np
from cap_from_youtube import cap_from_youtube

import tapnet.utils as utils
from tapnet.tapir_inference import TapirInference

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    input_size = 480
    num_points = 1024
    num_iters = 4  # Use 1 for faster inference, and 4 for better results

    # Initialize video
    # cap = cv2.VideoCapture("input.mp4")
    videoUrl = 'https://youtu.be/lOSgQxG_uqQ?si=QN8YiTFWbD4wnsS5'
    cap = cap_from_youtube(videoUrl, resolution="1080p")
    start_time = 0 # skip first {start_time} seconds
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))

    # Initialize model
    tapir = TapirInference('models/causal_bootstapir_checkpoint.pt', (input_size, input_size), num_iters, device)

    # Initialize query features
    query_points = utils.sample_grid_points(input_size, input_size, num_points)
    point_colors = utils.get_colors(num_points)
    ret, frame = cap.read()
    tapir.set_points(frame, query_points)

    # Reset video to the beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_time* cap.get(cv2.CAP_PROP_FPS))

    # out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))

    track_length = 30
    tracks = np.zeros((num_points, track_length, 2), dtype=object)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run the model
        points, visibles = tapir(frame)

        # Record visible points
        tracks = np.roll(tracks, 1, axis=1)
        tracks[visibles, 0] = points[visibles]
        tracks[~visibles, 0] = -1

        # Draw the results
        frame = utils.draw_tracks(frame, tracks, point_colors)
        frame = utils.draw_points(frame, points, visibles, point_colors)
        cv2.imshow('frame', frame)

        # out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
