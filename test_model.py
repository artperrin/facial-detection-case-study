# import the necessary packages
from assets.face_alignment import face_alignment
from sklearn.utils import shuffle
from collections import Counter
from imutils import paths
import config
import numpy as np
import logging as lg
import time
import argparse
import imutils
import pickle
import cv2
import os

lg.getLogger().setLevel(lg.INFO)

# filter by the size of the detected face
wh_filter = config.SIZE_DETECTION_THRESHOLD

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input test image")
ap.add_argument(
    "-r",
    "--recognizer",
    default="./output/recognizer.pickle",
    help="path to model trained to recognize faces (default ./output folder)",
)
ap.add_argument(
    "-l",
    "--le",
    default="./output/le.pickle",
    help="path to label encoder (default ./output folder)",
)
ap.add_argument(
    "-c",
    "--confidence",
    type=float,
    default=0.6,
    help="minimum probability to filter weak detections (default .6)",
)
ap.add_argument(
    "-v",
    "--visualization",
    type=bool,
    default=False,
    help="if the user wants to visualize each results as they are processed (default False)",
)
ap.add_argument(
    "-e",
    "--export",
    type=str,
    default=".",
    help="path to result log if wanted (main folder by default, None if not wanted)",
)
args = vars(ap.parse_args())

if args["export"] is not None:
    FILE = [f"""# Results log \n Test ran with confidence {args["confidence"]}. \n"""]
    FILE.append("")

start = time.time()
# load our serialized face detector from disk
lg.info("Loading face detector...")
detectorPath = config.DETECTOR
protoPath = os.path.sep.join([detectorPath, "deploy.prototxt"])
modelPath = os.path.sep.join([detectorPath, "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
# load our serialized face embedding model from disk
lg.info("Loading face recognizer...")

embeddings_model = config.EMBEDDINGS

if embeddings_model == "keras":
    from assets import model_bluider as mb
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.applications.imagenet_utils import preprocess_input
    from tensorflow.keras.models import Model
    import tensorflow.keras.backend as K
    import tensorflow as tf

    # to compute with GPU if available
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    model = mb.build_model()
    vgg_face = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
else:
    embedder = cv2.dnn.readNetFromTorch(embeddings_model)

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
Errors = []
positive_confidence = 0

# testing dataset description
lg.info("Describing dataset...")
classes = [p.split(os.path.sep)[-2] for p in imagePaths]
distinct = Counter(classes).keys()

for cl in distinct:
    lg.info(
        f"There are {sum([1 if name==cl else 0 for name in classes])} elements in the {cl} class."
    )

imagePaths = shuffle(imagePaths)

lg.info(f"Opening {len(imagePaths)} images and making predictions...")
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

            if args["visualization"]:
                cv2.imshow(nameTrue, cv2.resize(face, (350, 500)))
                cv2.waitKey(1)

            faceCopy = face.copy()
            face = face_alignment(face, visu=False)

            if embeddings_model == "keras":
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = np.expand_dims(face, axis=0)
                face = preprocess_input(face)
                face_encode = vgg_face(face)
                pred = np.squeeze(K.eval(face_encode))
                pred = pred.reshape(1, -1)
            else:
                # construct a blob for the face ROI, then pass the blob
                # through our face embedding model to obtain the 128-d
                # quantification of the face
                faceBlob = cv2.dnn.blobFromImage(
                    face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False
                )
                embedder.setInput(faceBlob)
                pred = embedder.forward()

            # perform classification to recognize the face
            if count == 1:  # if this is the first prediction
                try:
                    preds = recognizer.predict_proba(pred)[0]
                except ValueError as er:
                    lg.error(er)
                    lg.error(
                        f"It seems that the model has been trained to work with embeddings extrated with another method than {config.EMBEDDINGS},\n try to re-train your classifier or extract new embeddings with the expected method."
                    )
            else:
                preds = recognizer.predict_proba(pred)[0]
            j = np.argmax(preds)
            proba = preds[j]
            namePredict = le.classes_[j]
            NB_FACES += 1

            # test if the model is sure enough, otherwise class the person as unknow
            if proba <= args["confidence"] and namePredict != "unknow":
                namePredict = "unknow"

            # test if the result is a true positive/negative or a false positive/negative
            if namePredict == nameTrue:
                if namePredict == "unknow":
                    TN += 1
                else:
                    TP += 1
                    positive_confidence += proba
            else:
                if nameTrue == "unknow":
                    FP += 1
                else:
                    FN += 1

            if args["export"] is not None:
                if namePredict == nameTrue:
                    line = "SUCCESS -"
                else:
                    line = "ERROR   -"
                line += f"Original {nameTrue} | Predicted {namePredict} with confidence {proba} on file {imageName}, \n"
                FILE.append(line)

    if args["visualization"] is not None:
        cv2.destroyAllWindows()

    # draw a progress line
    progress = ""
    for k in range(30):  # length of the progress line
        if k <= int((count) * 30 / len(imagePaths)):
            progress += "="
        else:
            progress += "."

    print(f"{count}/{len(imagePaths)} [" + progress + "]", end="\r", flush=True)
    count += 1

print("", flush=True)

# conclusion
if NB_FACES < len(imagePaths):
    lg.warning(
        f"Not all faces have been detected, {len(imagePaths)-NB_FACES+1} missing."
    )

lg.info(
    f"Results of the test: {round(TP/(TP+FN)*100,1)} % true positives with {round(positive_confidence/TP, 2)} mean confidence,"
)
lg.info(f"-------------------: {round(TN/(TN+FP)*100,1)} % true negatives,")
lg.info(f"-------------------: {round(100-TN/(TN+FP)*100,1)} % false positives,")
lg.info(f"-------------------: {round(100-TP/(TP+FN)*100,1)} % false negatives.")
lg.info(f"Overall accuracy of the model : {round((TP+TN)/NB_FACES*100,1)} %.")

if args["export"] is not None:
    FILE[
        1
    ] = f"Number of errors: {FP+FN}/{NB_FACES}, overall accuracy {round((TP+TN)/NB_FACES*100,1)} %.\n"
    with open(args["export"] + "/results.log", "w") as file:
        file.writelines(FILE)

lg.info(f"Program ended within {round(time.time() - start, 2)} seconds.")
