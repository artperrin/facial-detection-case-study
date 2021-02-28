# import the necessary packages
from imutils import paths
from assets.face_alignment import face_alignment
from collections import Counter
from sklearn.utils import shuffle
import logging as lg
import numpy as np
import config
import argparse
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
ap.add_argument(
    "-i", "--dataset", required=True, help="path to input directory of faces + images"
)
ap.add_argument(
    "-e",
    "--embeddings",
    default="./output/embeddings.pickle",
    help="path to output serialized db of facial embeddings",
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
lg.info("Loading face detector...")
detectorPath = config.DETECTOR
protoPath = os.path.sep.join([detectorPath, "deploy.prototxt"])
modelPath = os.path.sep.join([detectorPath, "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
lg.info("loading face recognizer...")

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

# grab the paths to the input images in our dataset
lg.info("Quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))
# initialize our lists of extracted facial embeddings and
# corresponding people names
knownEmbeddings = []
knownNames = []
# initialize the total number of faces processed
total = 0

# training dataset description
lg.info("Describing dataset...")
classes = [p.split(os.path.sep)[-2] for p in imagePaths]
distinct = Counter(classes).keys()

for cl in distinct:
    lg.info(
        f"There are {sum([1 if name==cl else 0 for name in classes])} elements in the {cl} class."
    )


lg.info("Processing images...")
imagePaths = shuffle(imagePaths)
# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    name = imagePath.split(os.path.sep)[-2]
    # load the image, resize it to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image
    # dimensions
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
    # ensure at least one face was found
    if len(detections) > 0:
        # we're making the assumption that each image has only ONE
        # face, so find the bounding box with the largest probability
        j = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, j, 2]
        # ensure that the detection with the largest probability also
        # means our minimum probability test (thus helping filter out
        # weak detections)
        if confidence > 0.6:
            # compute the (x, y)-coordinates of the bounding box for
            # the face
            box = detections[0, 0, j, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # extract the face ROI and grab the ROI dimensions
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
            # ensure the face width and height are sufficiently large
            if fW < wh_filter or fH < wh_filter:
                continue

            if args["visualization"]:
                cv2.imshow("current processed image", face)
                cv2.waitKey(1)

            # pre-process the image by running face alignment
            face = face_alignment(face, False)

            if embeddings_model == "keras":
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = np.expand_dims(face, axis=0)
                face = preprocess_input(face)
                face_encode = vgg_face(face)
                pred = np.squeeze(K.eval(face_encode))
                knownEmbeddings.append(pred)
            else:
                # construct a blob for the face ROI, then pass the blob
                # through our face embedding model to obtain the 128-d
                # quantification of the face
                faceBlob = cv2.dnn.blobFromImage(
                    face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False
                )
                embedder.setInput(faceBlob)
                vec = embedder.forward()
                knownEmbeddings.append(vec.flatten())

            # add the name of the person + corresponding face
            # embedding to their respective lists
            knownNames.append(name)
            total += 1

    # draw a progress line
    progress = ""
    for k in range(30):  # length of the progress line
        if k <= int((i + 1) * 30 / len(imagePaths)):
            progress += "="
        else:
            progress += "."

    print(f"{i+1}/{len(imagePaths)} [" + progress + "]", end="\r", flush=True)

    if args["visualization"]:
        cv2.destroyAllWindows()

print("", end="\n", flush=True)

# dump the facial embeddings + names to disk
lg.info(f"Serializing {total} encodings...")
data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open(args["embeddings"], "wb")
f.write(pickle.dumps(data))
f.close()

lg.info(f"Program ended within {round(time.time()-start,2)} seconds.")
