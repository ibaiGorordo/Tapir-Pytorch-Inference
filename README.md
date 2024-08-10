# Tapir Pytorch Inference

![!Tapir Pytorch Inference](https://github.com/ibaiGorordo/Tapir-Pytorch-Inference/raw/main/doc/img/tapir_video.gif)

## Important
This is a strip down version of the original Tapir repository focused on inference.
- Removed the JAX dependencies and the training code.
- Converted tensors from 5D to 4D (only use one frame)
- Create models to simplify ONNX export
- Add onnx export script

## Installation
```shell
git clone https://github.com/ibaiGorordo/Tapir-Pytorch-Inference.git
cd Tapir-Pytorch-Inference
pip install -r requirements.txt
```

## ONNX Export
```shell    
python export_onnx.py
```

Arguments:
 - **--model**: Path to the model weights
 - **--resolution**: Input resolution (default: 640)
 - **--num_points**: Number of points (default: 1000)
 - **--num_iters**: Number of iterations, use 1 for faster inference, 4 for better results (default: 4)
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

# References:
* **TAPIR Repository:** [https://github.com/google-deepmind/tapnet/tree/main](https://github.com/google-deepmind/tapnet/tree/main)
