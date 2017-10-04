import numpy as np
import time
import cv2

def detectWithCaffe(args, input):
    '''neural network loading some settings and making predictions'''
    rows = open(args["labels"]).read().strip().split("\n")
    classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
    blob = cv2.dnn.blobFromImage(input, 1, (224, 224), (104, 117, 123))
    print("[INFO] Loading model ...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
    net.setInput(blob)
    start = time.time()
    preds = net.forward()
    end = time.time()
    print("[INFO] Classification took {:.5} seconds".format(end - start))
    idxs = np.argsort(preds[0])[::-1][:5]

    #loop over top 5 predictions and display them
    for (i, idx) in enumerate(idxs):
        #draw the top prediction on the input image
        if i == 0:
            text = "Label: {}, {:.2f}%".format(
                classes[idx],
                preds[0][idx] * 100,
            )
            cv2.putText(
                input,
                text,
                (5, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255), 2,
            )

        #display the predicted label and + probability to the console
        print(
            "[INFO] {}. label: {}, probability: {:.5}".format(
                i + 1,
                classes[idx],
                preds[0][idx],
            )
        )

    '''end neural network getting ready'''
    pass
