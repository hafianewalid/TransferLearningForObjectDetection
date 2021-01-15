# Transfer Learning For Object Detection
In this work we explore domain adaptation transfer learning with the frozen strategy. For single and multiple object detection tasks. We experiment several pretrained model as features extractor. A simple feedforward network has been used as features classifier for a single object detection and a Fully Convolutional network with YOLO approach in multiple object detection case. We perform those models on Pascal VOC 2012 dataset, and we test them in inferece for real time object detection.

## Data Exploration
In order to get a general idea about Pascal VOC 2012 objects, We have implemented a simple statistical data exploration methods. To show this stats run the following line command:

`$python data_exploration.py`

The statistics results should look like the following:

<img src="https://drive.google.com/uc?export=view&id=10fyE-IRpl6HDX50cNy1vvREPWBq34FQ_" width="500" height="500">
![Alt Text](https://drive.google.com/uc?export=view&id=10fyE-IRpl6HDX50cNy1vvREPWBq34FQ_)

## Single Object Detection 
### Features Extraction
There are several pretrained models we can use it as features extractor. The features will be saved in a specific format according to the task, single or multiple object. Those features will feed subseqencely an FeedForWard neural network.
To extract the features from Pascal VOC datasets using a pretrained model you can use the following command:

`$python features_extraction.py --pretrained resnet18 --model_type True`

### Training The Additional Parameters 
After the feature extraction phase, you can train the additional parameters by feeding them by the extracted features.

The following command allows to train the head nuralnetwork with the according training parameters:

`$python main.py --model_type True --lr 0.001 --num_ep 50`

To show the training graphs run the following command:

`$tensorboard --logdir logs/Experience_0`

The training ghraph for resnet18 should look like below: 

<img src="https://drive.google.com/uc?export=view&id=1V8jezR4TncmI-fquDes72AWHxno5XJ3Y" width="600" height="300">


### Results 
The table below shows the obtained results for single object detection with different feature extractor models and with FFW classifier:

| pretrained model     |  accurecy  |  fps(GPU)  |
|----------------------|------------|------------|
| resnet18             |  79.73     |  184       |
| resnet34             |  81.22     |  178       |
| resnet50             |  82.24     |  84        |
| resnet101            |  83.08     |  60        |
| resnet152            |  83.17     |  36        |
| alexnet              |  67.40     |  303       |
| vgg16                |  78.79     |  243       |
| shufflenet           |  74.06     |  95        |

### Inference 
To get the inference prediction for an image or set of images we can use the following commands:

`$python inference.py --model_file logs/Experience_0/best_model.pt --model_type True --image_file images/im1.jpg` 

`$python inference.py --model_file logs/Experience_0/best_model.pt --model_type True --images_set images`

#### Examples:

<img src="https://drive.google.com/uc?export=view&id=190XS_KJ0PPk97OMoWvFtDkfNfajQ3mbv">

To get the heat-map associated to the prediction use the following command:

`$python inference.py --model_file logs/Experience_0/best_model.pt --model_type True --image_file images/im1.jpg --heat_map True`

#### Examples:

<img src="https://drive.google.com/uc?export=view&id=1qRrsQAw3EaYFpvU-ac_4kr-QnmNvaCiq">

To run real time inference using a cam device or a video stream you can run the following commands:

`$python inference.py --model_file logs/Experience_0/best_model.pt --model_type True --stream True --cam 0`

`$python inference.py --model_file logs/Experience_0/best_model.pt --model_type True --stream True --video video.mp4`


### Real Time Demonstration

![Alt Text](https://media.giphy.com/media/hVSL3Xa4vFgKK2a9sW/giphy.gif)

## Multiple Objects Detection 
### Features Extraction
To extract the features from Pascal VOC datasets using a pretrained model you can use the following command:

`$python features_extraction.py --pretrained resnet18`

### Training The Additional Parameters 
To train the additional parameters run the following command:

`$python main.py --lr 0.001 --num_ep 50`

The training ghraph for resnet18 should look like below: 

<img src="https://drive.google.com/uc?export=view&id=1G9AsquL4tSJYmE-9y0L27KvZvZE66Q4k" width="600" height="300">

### Results 
The table below shows the obtained results for multiple objects detection with different feature extractor models and with FCN classifier:

| pretrained model     |  accurecy  |  fps(GPU)  |
|----------------------|------------|------------|
| resnet18             |  68.09     |  83        |
| resnet34             |  72.07     |  67        |
| resnet50             |  73.93     |  57        |
| resnet101            |  74.92     |  43        |
| resnet152            |  75.67     |  32        |
| vgg16                |  68.48     |  96        |
| shufflenet           |  66.26     |  58        |

### Inference 

To use the inference for mutiple object detection, run the same comands as single object without model_type option :

`$python inference.py --model_file logs/Experience_0/best_model.pt --image_file images/im1.jpg `

`$python inference.py --model_file logs/Experience_0/best_model.pt --image_file images/im1.jpg --heat_map True`

`$python inference.py --model_file logs/Experience_0/best_model.pt --images_set images`

`$python inference.py --model_file logs/Experience_0/best_model.pt --stream True --cam 0`

`$python inference.py --model_file logs/Experience_0/best_model.pt --stream True --video video.mp4`


#### Exemple :

<img src="https://drive.google.com/uc?export=view&id=1fJl3XHXaY3TKcrDIPo4B8y65TgYhqXvx">

### Real Time Demonstration

![Alt Text](https://media.giphy.com/media/JTwrTGHkAdwtxOUJCk/giphy.gif)
