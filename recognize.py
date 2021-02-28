# import the necessary packages
from assets.face_alignment import face_alignment
import numpy as np
import logging as lg
import argparse
import config
import imutils
import pickle
import cv2
import os
import time

lg.getLogger().setLevel(lg.INFO)

# filter by the size of the detected face
wh_filter = config.SIZE_DETECTION_THRESHOLD

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
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
    default=0.7,
    help="minimum probability to filter weak detections",
)
ap.add_argument(
    "-v",
    "--visualization",
    type=bool,
    default=False,
    help="if the user wants to visualize the face alignment",
)
args = vars(ap.parse_args())

start = time.time()
# load our serialized face detector from disk
detectorPath = config.DETECTOR
protoPath = os.path.sep.join([detectorPath, "deploy.prototxt"])
modelPath = os.path.sep.join([detectorPath, "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
# load our serialized face embedding model from disk
lg.info("Loading face recognizer...")

embeddings_model = config.EMBEDDINGS

if embeddings_model == "keras":
    from assets import model_bluider as mb
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
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
image = cv2.imread(args["image"])
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
    if confidence > 0.5:
        # compute the (x, y)-coordinates of the bounding box for the
        # face
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        # extract the face ROI
        face = image[startY:endY, startX:endX]
        (fH, fW) = face.shape[:2]
        # ensure the face width and height are sufficiently large
        if fW < wh_filter or fH < wh_filter:
            continue

        face = face_alignment(face, args["visualization"])
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

        preds = recognizer.predict_proba(pred)[0]
        j = np.argmax(preds)
        proba = preds[j]
        name = le.classes_[j]

        # test if the model is sure enough, otherwise class the person as unknow
        if proba <= args["confidence"]:
            name = "unknow"

        # draw the bounding box of the face along with the associated
        # probability
        text = "{}: {:.2f}%".format(name, proba * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(
            image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2
        )

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imwrite("output_image.jpg", image) # uncomment this to save the end image

lg.info(f"Program ended within {round(time.time()-start, 2)} seconds.")
