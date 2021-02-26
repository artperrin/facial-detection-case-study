# import the necessary packages
from assets.face_alignment import face_alignment
from collections import Counter
from imutils import paths
import numpy as np
import logging as lg
import argparse
import imutils
import pickle
import cv2
import os

lg.getLogger().setLevel(lg.INFO)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input test image")
ap.add_argument(
    "-d",
    "--detector",
    default="./assets/face_detection_model",
    help="path to OpenCV's deep learning face detector",
)
ap.add_argument(
    "-m",
    "--embedding-model",
    default="./assets/openface.nn4.small2.v1.t7",
    help="path to OpenCV's deep learning face embedding model",
)
ap.add_argument(
    "-r",
    "--recognizer",
    default="./output/recognizer.pickle",
    help="path to model trained to recognize faces",
)
ap.add_argument(
    "-l", "--le", default="./output/le.pickle", help="path to label encoder"
)
ap.add_argument(
    "-c",
    "--confidence",
    type=float,
    default=0.5,
    help="minimum probability to filter weak detections",
)
ap.add_argument(
    "-v",
    "--visualization",
    type=bool,
    default=False,
    help="if the user wants to visualize each results",
)
args = vars(ap.parse_args())

# load our serialized face detector from disk
lg.info("Loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join(
    [args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"]
)
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
# load our serialized face embedding model from disk
lg.info("Loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])
# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

lg.info("Loading image and applying detection...")
# load the image, resize it to have a width of 600 pixels (while
# maintaining the aspect ratio), and then grab the image dimensions
testPath = args["image"]
imagePaths = list(paths.list_images(testPath))

count = 1
NB_FACES = 0

TP = 0
TN = 0
FP = 0
FN = 0

# testing dataset description
lg.info("Describing dataset...")
classes = [p.split(os.path.sep)[-2] for p in imagePaths]
distinct = Counter(classes).keys()

for cl in distinct:
    lg.info(
        f"There are {sum([1 if name==cl else 0 for name in classes])} elements in the {cl} class."
    )

lg.info("Opening images and making predictions...")
# loop over the image paths
for imageName in imagePaths:
    image = cv2.imread(imageName)
    nameTrue = imageName.split(os.path.sep)[-2]
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
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # extract the face ROI
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            if args["visualization"]:
                cv2.imshow(nameTrue, cv2.resize(face, (350, 500)))
                cv2.waitKey(1)

            face = face_alignment(face, visu=False)
            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(
                face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False
            )
            embedder.setInput(faceBlob)
            vec = embedder.forward()
            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            namePredict = le.classes_[j]
            NB_FACES += 1

            # test if the result is a true positive/negative or a false positive/negative
            if namePredict == nameTrue:
                if namePredict == "unknow":
                    TN += 1
                else:
                    TP += 1
            else:
                if nameTrue == "unknow":
                    FP += 1
                else:
                    FN += 1

    status = f"Image {count} out of {len(imagePaths)}."

    if args["visualization"]:
        cv2.destroyAllWindows()

    count += 1
    print(status, end="\r", flush=True)

print("", flush=True)

# conclusion
if NB_FACES < len(imagePaths):
    lg.warning(
        f"Not all faces have been detected, {len(imagePaths)-NB_FACES+1} missing."
    )

lg.info(f"Results of the test: {round(TP/NB_FACES*100,1)} % true positives,")
lg.info(f"-------------------: {round(TN/NB_FACES*100,1)} % true negatives,")
lg.info(f"-------------------: {round(FP/NB_FACES*100,1)} % false positives,")
lg.info(f"-------------------: {round(FN/NB_FACES*100,1)} % false negatives.")
lg.info(
    f"Overall accuracy of the model : {round(TP/NB_FACES*100,1)+round(TN/NB_FACES*100,1)} %."
)
