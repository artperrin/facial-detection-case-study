# Case study : facial detection and recognition with Python

This repository contains a work made for a Computer Vision course at the *Ecole Centrale de Lille*.

Its aim is to detect and recognize each face, firstly of an image, then of a video.
As an example, I used pictures of celebrities.

![example](./readme_figures/output.jpg)

This repository contains:
* a extract_embeddings.py script to extract embeddings from the dataset,
* a train_model.py script to train the SVM model to classify the subjects,
* a recognize.py script to run the model on an image,
* a test_model.py script to run the model on multiple images (almost strict copy of the previous script).

First of all, the user must install the needed libraries by running the command line:
```sh
$ pip install -r requirements.txt
```

Each of the scripts are to be runned using the command line:
```sh
$ python path/to/script.py --args
```

and all the arguments are described with the command line:
```sh
$ python path/to/script.py --help
```

## News

In this section I will write the different steps of my work.

### Firstly 

I started with a simple database containing 30 pictures of Emma Stone, as much of Ryan Gosling, and 30 pictures of different celebrities labeled as "unknown". The extract_embeddings.py script firstly creates embeddings, which are vectors describing the "face's properties", then a Support Vector Machine model is trained to classify the celebrities.
It shows some good results (see the example above), but there are some errors : the results seem to be quite sensitive to the face's alignment.

![face alignment problem](./readme_figures/face_alignment_problem.jpg)

Moreover, the model seems to struggle more with female faces, as quite a lot of female "unknown" faces are classified as Emma Stone's.

![false stone detected](./readme_figures/false_stone.jpg)

In the future, I shall expand my database for each class and implement a face alignment script to tackle the related trouble.

### Secondly

After having implemented a face_alignment.py program, I trained the model and run the tests on a total of 45 images:

|True positive|True negative|False positive|False negative|
| :----- | :----: | :----: | ------:|
| 28.8 % | 37.8 % | 17.8 % | 15.6 % |

As the results seems to be better, I've encountered some unusual behaviour: some female faces have been detected as Ryan Gosling's.

![false positive](./readme_figures/weird_false_positive.jpg)

I will now search for more data to train the model and run another test.

***Edit: My testing program was wrong, because the input faces were not aligned. I corrected this and the above false result disappeared:***

![corrected](./readme_figures/fixed.jpg)

## Credits

This code has been implemented based on [this tutorial](https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/) written by [Adrian Rosebrock](https://github.com/jrosebr1), whose very good content has been quite helpful for me.