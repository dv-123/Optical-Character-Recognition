# Optidash_Project_OCR

## What is an OCR ? 
Optical character recognition or optical character reader is the electronic or mechanical conversion of images of typed, handwritten or printed text into machine-encoded text, whether from a scanned document, a photo of a document, a scene-photo or from subtitle text superimposed on an image.

![](Displayimages/image_1.jpeg)


To train our network to recognize these sets of characters, we utilized the MNIST digits dataset as well as the NIST Special Database 19 (for the A-Z characters).

### Installation
First, clone or download this GitHub repository. Install requirements and download pretrained weights:

```
pip install -r ./requirements.txt

# yolov3
wget -P model_data https://pjreddie.com/media/files/yolov3.weights

# yolov3-tiny
wget -P model_data https://pjreddie.com/media/files/yolov3-tiny.weights

# yolov4
wget -P model_data https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

# yolov4-tiny
wget -P model_data https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
```
## Quick training for custom mnist dataset
mnist folder contains mnist images, create training data:
```
python mnist/make_data.py
```
`./yolov3/configs.py` file is already configured for mnist training.

The requirements.txt file will include all the needed libraries for the project.

![](Displayimages/Requirements.jpeg)

Open the terminal and type in 
```
mnist/make_data.py
```
This will create all training data needed for the project and it converts the data.

To train the model use 
```
python train.py
```
One can train this model on GPU for faster results.



A little about the dataste used for taining:-

The mnist folder has the subfolder named mnist , which contains the zip files of mnist training dataset i.e. test and train folders, which is extracted to the mnist_test and mnist_train folders.
The noise_image folder contains the background images (i.e. pictures of the documents) for the training dataset which we have been made by us.






















Now, you can train it and then evaluate your model
```
python train.py
tensorboard --logdir=log
```
Track training progress in Tensorboard and go to http://localhost:6006/:

<p>
    <img width="50%" src="Displayimages/Tensorflow_1.jpeg" style="max-width:100%;"></a>
</p>

<p>
    <img width="50%" src="Displayimages/Tensorflow_2.jpeg" style="max-width:100%;"></a>
</p>

<p>
    <img width="50%" src="Displayimages/Tensorflow_3.jpeg" style="max-width:100%;"></a>
</p>

<p>
    <img width="50%" src="Displayimages/Tensorflow_4.jpeg" style="max-width:100%;"></a>
</p>

<p>
    <img width="50%" src="Displayimages/Tensorflow_5.jpeg" style="max-width:100%;"></a>
</p>


