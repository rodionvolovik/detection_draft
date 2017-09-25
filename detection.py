import sys
import cv2
import argparse
from imageprocessor import detect

'''
for detecting from built-in web-camera
'''
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--camera", required=False, help="[port of camera for input video stream]")
ap.add_argument("-i", "--image", required=False, help="[path to input image]")
args = vars(ap.parse_args())

print(args)

if 'camera' in args.keys() and args['camera']:
    src = int(args['camera'])
    input = cv2.VideoCapture(src)

    ret = True
    while ret:
        ret, image = input.read()
        detect()
        try:
            cv2.imshow("input", image)
        except:
            print("Some problems with parameters")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if 'image' in args.keys() and args['image']:
    input = cv2.imread(args['image'])
    detect()
    cv2.imshow("input", input)
    cv2.waitKey(3000)