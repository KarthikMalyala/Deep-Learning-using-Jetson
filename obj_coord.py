import jetson.inference
import jetson.utils

import cv2
import numpy as np
from time import sleep
import pyrealsense2 as rs
import numpy as np
import math

width = 640
height = 480
fps = 30

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
config.enable_stream(rs.stream.depth, width, height, rs.format.z16,  fps)

profile = pipeline.start(config)

net = jetson.inference.detectNet(argv=['--model=python/training/detection/ssd/models/learn/ssd-mobilenet.onnx', '--labels=python/training/detection/ssd/models/learn/labels.txt', '--input-blob=input_0', '--output-cvg=scores', '--output-bbox=boxes'])

speed = 0.0
cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

focalLengthX = width/(2*(math.tan((87/2) * (math.pi/180))))
focalLengthY = height/(2*(math.tan((58/2) * (math.pi/180))))

while True:
	frames = pipeline.wait_for_frames()
	
	color_frame = frames.get_color_frame()	
	depth_frame = frames.get_depth_frame()



	depth_image = np.asanyarray(depth_frame.get_data())
	color_image = np.asanyarray(color_frame.get_data())

	frame = np.asanyarray(color_frame.get_data())
	cuda_mem = jetson.utils.cudaFromNumpy(frame)
	detections = net.Detect(cuda_mem)

	cv2.waitKey(1)	
	overlay = color_image.copy()
	output = color_image.copy()
	  
	# font
	font = cv2.FONT_HERSHEY_SIMPLEX
	  
	# org
	org = (50, 50)
	  
	# fontScale
	fontScale = 1
	   
	# Blue color in BGR
	color = (255, 0, 0)
	  
	# Line thickness of 2 px
	thickness = 2

	for detection in detections:
		# Prints the name of the object
		#print(net.GetClassDesc(detection.ClassID))

		# Prints the entire detection details
		print(detection)

		imgCentrX = width/2
		imgCentrY = height/2
		imgDistX = int(detection.Center[0]) - imgCentrX
		imgDistY = imgCentrY - int(detection.Center[1])

		realDepthZ = depth_frame.get_distance(int(detection.Center[0]), int(detection.Center[1]))		
		realDistX = (realDepthZ * imgDistX) / int(focalLengthX)
		realDistY = (realDepthZ * imgDistY) / int(focalLengthY)

		#if(realDepthZ != 0.0 and realDistX != 0.0 and realDistY != 0.0 and realDepthZ <= 1):
		print('\n')			
		#print('X: ', realDistX)
		#print('Y: ', realDistY)
		#print('Z: ', realDepthZ)

		cv2.rectangle(overlay, (int(detection.Left), int(detection.Top),), (int(detection.Right), int(detection.Bottom)),(0, 0, 255), -1)
		cv2.putText(output, str(realDistX), org, font, fontScale, color, thickness)
		cv2.addWeighted(overlay, 0.5, output, 0.5,0, output)

		sleep(0.1)

	cv2.imshow('RealSense', output)


























