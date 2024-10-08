# Tapir Pytorch Inference
https://github.com/user-attachments/assets/457eeb57-9961-4022-9b15-55f1d9dc2260

## Important
This is a strip down version of the original Tapir repository focused on inference.
- Removed the JAX dependencies and the training code.
- Make it easy to run in real-time even with a camera feed.
- Converted tensors from 5D to 4D (only use one frame)
- ⚠️⚠️⚠️**ONNX Inference is very slow**⚠️⚠️⚠️

## Installation
```shell
git clone https://github.com/ibaiGorordo/Tapir-Pytorch-Inference.git
cd Tapir-Pytorch-Inference
pip install -r requirements.txt
```
- **Download model** from: https://storage.googleapis.com/dm-tapnet/causal_bootstapir_checkpoint.pt

## License
The License of the original model is Apache 2.0: [License](https://github.com/google-deepmind/tapnet/blob/main/LICENSE)

## ONNX Export
⚠️⚠️⚠️**ONNX Inference is very slow**⚠️⚠️⚠️
```shell    
python onnx_export.py
```

Arguments:
 - **--model**: Path to the model weights
 - **--resolution**: Input resolution (default: 640)
 - **--num_points**: Number of points (default: 1000)
 - **--dynamic**: Export with dynamic number of points (default: False)
 - **--num_iters**: Number of iterations, use 0 for faster inference, 4 for better results (default: 4)
 - **--output_dir**: Output directory (default: ./)

# Examples
## **Video inference**:

 ```shell
 python example_video_tracking.py
 ```

## **Webcam inference**:

 ```shell
 python example_webcam_tracking.py
 ```

## References:
* **TAPIR Repository:** [https://github.com/google-deepmind/tapnet/tree/main](https://github.com/google-deepmind/tapnet/tree/main)
