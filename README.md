# Deep-Learning-using-Jetson

## Video Demonstration: https://youtu.be/ecxtygfc0DE

### To run using the LARRINX NVIDIA Jetson NX (LARRI Personnel Only):
- cd jetson-inference
- docker/run.sh
- cd python/training/detection/ssd
- detectnet --model=models/learn/ssd-mobilenet.onnx --labels=models/learn/labels.txt --input-blob=input_0 --output-cvg=scores --output-bbox=boxes /dev/video2

### Setup for any other Jetson device:
- Download the image training files (JPEGs, Annotations, and Sets) using this link: https://louisville.box.com/s/4563epmivg19conll2aotole2tq7abuj
- Clone the repository from NVIDIA: https://github.com/dusty-nv/jetson-inference.git
- Under jetson-inference/python/training/detection/ssd/data, add the downloaded 'learn' folder to allow for image training

#### To train the above condiment 'learn' dataset:
- On the terminal:
  - cd jetson-inference
  - docker/run.sh
  - cd python/training/detection/ssd
  - python3 train_ssd.py --dataset-type=voc --data=data/learn --model-dir=models/learn --batch-size=2 --workers=1 --epochs=2
  
