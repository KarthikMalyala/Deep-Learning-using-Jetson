# Deep-Learning-using-Jetson

## Video Demonstration: https://youtu.be/ecxtygfc0DE

To run using the LARRINX NVIDIA Jetson NX:
- cd jetson-inference
- docker/run.sh
- cd python/training/detection/ssd
- detectnet --model=models/learn/ssd-mobilenet.onnx --labels=models/learn/labels.txt --input-blob=input_0 --output-cvg=scores --output-bbox=boxes /dev/video2

For any other device:
- Download the image training files (JPEGs, Annotations, and Sets) using this link: https://louisville.box.com/s/4563epmivg19conll2aotole2tq7abuj
- Clone the repository from NVIDIA: https://github.com/dusty-nv/jetson-inference.git
- Under jetson-inference/python/training/detection/ssd, add the downloaded 'learn' folder to allow for image training
- 
