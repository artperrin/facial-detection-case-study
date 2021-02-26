# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import logging as lg
import argparse
import pickle
import time

lg.getLogger().setLevel(lg.INFO)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "-e",
    "--embeddings",
    default="./output/embeddings.pickle",
    help="path to serialized db of facial embeddings",
)
ap.add_argument(
    "-r",
    "--recognizer",
    default="./output/recognizer.pickle",
    help="path to output model trained to recognize faces",
)
ap.add_argument(
    "-l", "--le", default="./output/le.pickle", help="path to output label encoder"
)
args = vars(ap.parse_args())

start = time.time()
# load the face embeddings
lg.info("Loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())
# encode the labels
lg.info("Encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
lg.info("Training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# write the actual face recognition model to disk
f = open(args["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()
# write the label encoder to disk
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()

lg.info(f"Program ended within {round(time.time()-start,2)} seconds.")
