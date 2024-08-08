import cv2
import torch

import tapnet.utils as utils
from tapnet.tapir_inference import TapirInference

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    input_size = 480
    num_points = 1000
    num_iters = 4  # Use 1 for faster inference, and 4 for better results

    cap = cv2.VideoCapture('https://storage.googleapis.com/dm-tapnet/horsejump-high.mp4')

    tapir = TapirInference('models/causal_bootstapir_checkpoint.pt', (input_size, input_size), num_iters, device)

    # Initialize query features
    query_points = utils.sample_random_points(input_size, input_size, num_points)
    point_colors = utils.random.randint(0, 255, (num_points, 3))
    ret, frame = cap.read()
    tapir.set_points(frame, query_points)

    # Reset video to the beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run the model
        tracks, visibles = tapir(frame)

        # Draw the results
        frame = utils.draw_points(frame, tracks, visibles, point_colors)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
