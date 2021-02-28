from imutils import paths
from imutils import build_montages
import config
import logging as lg
import numpy as np
import cv2
import imutils
import os
import time
import argparse

wh_filter = config.SIZE_DETECTION_THRESHOLD

lg.getLogger().setLevel(lg.INFO)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to dataset to be tested")
ap.add_argument(
    "-v",
    "--visualization",
    type=bool,
    default=False,
    help="if the user wants to visualize the errors as they come",
)
args = vars(ap.parse_args())

lg.info("Loading detector and images...")
start = time.time()
errors = []
detectorPath = "./assets/face_detection_model"
protoPath = os.path.sep.join([detectorPath, "deploy.prototxt"])
modelPath = os.path.sep.join([detectorPath, "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

imagePaths = list(paths.list_images(args["dataset"]))
ct = 1
clean = True
lg.info("Beginning face detections...")
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]
    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0),
        swapRB=False,
        crop=False,
    )
    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()
    # loop over the detections
    total = 0
    FACES = []
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections
        if confidence > 0.6:
            # compute the (x, y)-coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # extract the face ROI
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
            # ensure the face width and height are sufficiently large
            if fW < wh_filter or fH < wh_filter:
                continue

            FACES.append(face.copy())
            print(f"Progress {ct} out of {len(imagePaths)}", end="\r", flush=True)
            total += 1
    ct += 1

    if total > 1:
        errors.append(f"\n ERROR file {imagePath}, {total} faces detected ! \n")
        clean = False
        if args["visualization"]:
            img = build_montages(FACES, (500, 500), (total, 1))[0]
            cv2.imshow(f"error in file {imagePath}", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

print("", end="\n")
lg.info("Face detections ended.")

if clean:
    lg.info("Dataset clean !")
else:
    lg.warning(
        "Dataset not clean, try to increase the weight/height filter or change the incriminated files:"
    )
    for e in errors:
        print(e)

lg.info(f"Program ended within {round(time.time()-start,2)} seconds.")
