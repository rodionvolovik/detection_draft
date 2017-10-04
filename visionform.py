from imageprocessor import detectWithCaffe
import cv2

def cameraVision(args):
    src = int(args['camera'])
    input = cv2.VideoCapture(src)

    ret = True
    while ret:
        ret, image = input.read()
        detectWithCaffe(args, image)
        try:
            cv2.imshow("input", image)
        except:
            print("Some problems with parameters")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def imageVision(args):
    input = cv2.imread(args['image'])
    detectWithCaffe(args, input)
    cv2.imshow("input", input)
    cv2.waitKey(0)