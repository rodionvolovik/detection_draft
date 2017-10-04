import sys
import argparse
from visionform import *

'''
for detecting from built-in web-camera
'''
ap = argparse.ArgumentParser(description="Object detection and recognition")
ap.add_argument("-c",
                "--camera",
                required=False,
                help="[port of camera for input video stream]",
                )
ap.add_argument("-i",
                "--image",
                required=False,
                help="[path to input image]",
                )
ap.add_argument("-p",
                "--prototxt",
                required=True,
                help="path to Caffe 'deploy' prototxt file",
                )
ap.add_argument("-m",
                "--model",
                required=True,
                help="path to Caffe pre-trained model",
                )
ap.add_argument("-l",
                "--labels",
                required=True,
                help="path to ImageNet labels (i.e., syn-sets)",
                )
ap.add_argument("-co", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
print("[INFO] Command line args voc. is : {}".format(args))

if 'camera' in args.keys() and args['camera']:
    cameraVision(args)

if 'image' in args.keys() and args['image']:
    imageVision(args)

