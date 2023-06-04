
# Hand Gesture AI Virtual Mouse Using Deep Learning Method
## Overview
In this project we want to make AI virtual mouse.So that users will have same user experience while using physical mouse in laptop or desktop.

## Machine Learning Pipeline

![image](https://github.com/Hem5555/AI-Hand-Gesture-mouse-using-Deep-Learning/assets/121716939/f428e4dc-f7c0-48b0-aa7f-5920787ce7a0)

Fig: our Machine Learning Pipeline for  virtual mouse
        
**Section 1:Problem Formulation:** Once we identified our problem to make user friendly  ai virtual mouse for daily use.

**Section 2: Collect and Label data**:Before collecting data, We install neccesary python libraries and modules. 


          Python 3.7 version (64 bit) on windows 10

          conda create -n aim python=3.7

          pip install mediapipe==0.8.1

          pip install opencv-python==3.4.2.17

          pip install tensorflow==2.4.0

          pip install scikit-learn==0.24.0

          pip install pynput

          pip install pyautogui

          pip install notebook
          pip install scikit-learn

          pip install pandas

          pip install seaborn


These are the following steps we follow to collect and label data.
* For Hand sign recognition training data collection, we Press "k" to enter the mode to save key points.

* Once we  press "0" to "9", the key points will be added to "model/keypoint_classifier/keypoint.csv".In the initial state, three types of learning data are included: open hand (class ID: 0), close hand (class ID: 1), and pointing (class ID: 2).

* Then we , added 4  classID chronologically to prepare the training data.

* Then. we change the number of training data classes, change the value of "NUM_CLASSES = 7" and modify the label of "model/keypoint_classifier/keypoint_classifier_label.csv" as appropriate.

*  we keep default option for Finger gesture recognition training.

* 75%:25% data is fragmented for training and testing data respectively.

* Altogether we collected more than 150K handgesture dataset.

* They have been categories under seveb class such as open,close,left click,right click,Pointer,scroll up and scroll down. 

![image](https://github.com/Hem5555/AI-Hand-Gesture-mouse-using-Deep-Learning/assets/121716939/92f6fe34-7018-4798-82a3-dc87b720fe85)

Fig 1: Hand Gesture Classification with class ID


**Section 3**: Evaluate data and Section 4:Feature engineering: We evaluate data whether they are repesenting well. 

**Section 5: Select and train model**: Once data collection is  completed, we Open "keypoint_classification.ipynb" in Jupyter Notebook and execute from top to bottom.

**Section 6: Deploy model**: We deploy our model in our code  whether these model can automate mouse with our input human gestures.

**Section 7: Evalute m0del**: Our model is evaluated based  on its accuracy,precision and recall on confusion matrix.Higher these values  will be considered as good model. We observe ourself are we able to meet our business goal.

**Section 8: Tune model**: We tune our model by changing hyperparameters.

### Gestures
Once we train our model, we are able to get following class under trained hand gestures.

![image](https://github.com/Hem5555/AI-Hand-Gesture-mouse-using-Deep-Learning/assets/121716939/44c4d304-025c-4687-b50f-d5fe94f81ca3)

![image](https://github.com/Hem5555/AI-Hand-Gesture-mouse-using-Deep-Learning/assets/121716939/e207f72b-ea03-4986-820f-e1a179711b97)


### Algorithms used for Hand Tracking
We are using mediapipeline in this project.MediaPipe framework is used for hand gesture and hand tracking and openCV library is used for computer vision.It is an opensource framework of Google. MediaPipe Hands is a high-fidelity hand and finger tracking solution. It employs machine learning (ML) to infer 21 3D landmarks of a hand from just a single frame.
The MediaPipe framework is based on three fundamental parts; they are performance evaluation, framework for retrieving sensor data, and a collection of components which are called calculators, and they are reusable. A pipeline is a graph which consists of components called calculators, where each calculator is connected by streams in which the packets of data flow through. Developers are able to replace or define custom calculators anywhere in the graph creating their own application.
### Architecture
Our hand tracking solution utilizes an ML pipeline consisting of two models working together.
* A palm detector that operates on a full input image and locates palms via an oriented hand bounding box.
* A hand landmark model that operates on the cropped hand bounding box provided by the palm detector and returns high-fidelity 2.5D landmarks. 
When the accurately cropped palm image are feeded to  hand landmark model, it drastically reduces the need for data augmentation (e.g. rotations, translation and scale) and allows the network to dedicate most of its capacity towards landmark localization accuracy.




![image](https://github.com/Hem5555/AI-Hand-Gesture-mouse-using-Deep-Learning/assets/121716939/53889b09-eac2-4d35-b92e-a8612b361a30)

                    
Fig 3: Hand perception pipeline overview
Source: https://ai.googleblog.com/2019/08/on-device-real-time-hand-tracking-with.html 



### Models
#### Palm Detection Model
To detect initial hand locations,a singleshot detector model is employed for optimized for mobile real-time application similar to BlazeFace which is also available in Mediapipe.Detecting hand is complex  to detect occluded and self-occluded hands. Similarly, faces have high contrast patterns, e.g., around the eye and mouth region.So, the lack of such features in hands makes it comparatively difficult to detect them reliably from their visual features alone.In order to solve this issues we overcome them from following solutions.
* First, palm detector is trained instead of a hand detector, since estimating bounding boxes of rigid objects like palms and fists is comparatively easier than detecting hands with articulated fingers. In addition, as palms are smaller objects, the non-maximum suppression algorithm performs well even for the two-hand self-occlusion cases such as handshakes. Further, palms can be modelled using only square bounding boxes, ignoring other aspect ratios, and therefore reducing the number of anchors by a factor of 3∼5.
* Second, we use an encoder-decoder feature extractor similar to FPN for a larger scene-context awareness even for small objects.
* Finally, we minimize the focal loss during training to support a large number of anchors resulting from the high scale variance.

![image](https://github.com/Hem5555/AI-Hand-Gesture-mouse-using-Deep-Learning/assets/121716939/ad26ba6a-4702-469a-9529-bf0d717a3b8f)

Figure 4: Palm detector model architecture

HandLand Mark Model
* Once the  palm detection are run over the whole image, our subsequent hand landmark model performs precise landmark localization of 21 2.5D coordinates inside the detected hand regions via regression. The model learns a consistent internal hand pose representation and is robust even to partially visible hands and self-occlusions. The model consists three outputs:

* 21 hand landmarks consisting of x, y, and relative depth. 
* A hand flag indicating the probability of hand presence in the input image. 
* A binary classification of handedness, e.g. left or right hand.


![image](https://github.com/Hem5555/AI-Hand-Gesture-mouse-using-Deep-Learning/assets/121716939/14ebbdac-3b72-4e3f-ab62-6116e8869ab7)
Figure5 : Architecture of our hand landmark model. The model has three outputs sharing a feature extractor. Each head is trained by correspondent datasets marked in the same color.

### Implementation of Mediapipeline
Using MediaPipe, our hand tracking pipeline can be built as a directed graph of modular components, called Calculators. Mediapipe comes with an extensible set of Calculators to solve tasks like model inference, media processing, and data transformations across a wide variety of devices and platforms. Individual Calculators like cropping, rendering and neural network computations are further optimized to utilize GPU acceleration.
Our MediaPipe graph for hand tracking is shown below. The graph consists of two subgraphs—one for hand detection and one for hand keypoints (i.e., landmark) computation. One key optimization MediaPipe provides is that the palm detector is only run as necessary (fairly infrequently), saving significant computation time. We achieve this by inferring the hand location in the subsequent video frames from the computed hand key points in the current frame, eliminating the need to run the palm detector over each frame.

![image](https://github.com/Hem5555/AI-Hand-Gesture-mouse-using-Deep-Learning/assets/121716939/d847536e-fac9-4c1c-9d03-dc5bd2e35c2e)
                            
Fig6: The hand landmark model’s output (REJECT_HAND_FLAG) controls when the hand detection model is triggered. This behavior is achieved by MediaPipe’s powerful synchronization building blocks, resulting in high performance and optimal throughput of the ML pipeline.


### MediaPipeline Hand marks
![image](https://github.com/Hem5555/AI-Hand-Gesture-mouse-using-Deep-Learning/assets/121716939/42154fb1-488d-46cb-ab8a-3bf55f70bdaf)
Fig 7: Hand landmark given by MediaPipe

### Demonstration
Once we execute the system and it starts capturing  video of webcam.It captures frames using WebCam. Once it detects our hands and hand tips through mediapipe and Open CV, it draws hand Landmarks and   rectangular box around the hand. On that rectangular box, we will play our mouse in this region of the PC window. Once the system detects our hands where its up or down. We will follow following condition.
* If our index finger is up, the mouse cursor will also move around the window where the index finger is located.This index finger also works as neural position as mouse.
* If index and middle finger is  raise up with distinct gap eachother, our  gesture will act mouse right button click
* Similarly, if we show index and finger meet eachother  or  we show ok sign, our hand gesture will work as mouse left click button.
* Further. If  we raise index and thumb finger up with certain distance  by closing other fingers. Then, our hand gesture will  scroll  Up function.
* Likewise, we raised only index, thumb and middle finger up, then  it will perform mouse scroll down function.
* Lastly, we can terminate  webcam by pressing esc button on our keyboard.


![image](https://github.com/Hem5555/AI-Hand-Gesture-mouse-using-Deep-Learning/assets/121716939/c7f8fe64-ad0b-43f8-8c3d-45d1a29bf5da)

Fig 8: Flowchart of the real-time AI virtual mouse system.


### Experimental Results and Evaluation

From our below confusion matrix, we accomplish 90% accuracy.We gain 0.98  precision,  0.72 recall and   0.83 f1 score for open hand gesture.Similarly, we obtain  0.85  precision,  0.87 recall and 0.86 f1 score for close hand gesture.In case of pointer, we get 0.85,0.95, 0.89 as precsion,recall and f1 score.Further, we score 0.99,0.96, 0.97 as precsion,recall and f1 score for left click hand gesture.In addition,  we receive 0.97 precision, 0.99 recall  and 0.98 f1 score for right click.For  scroll down,we achieve  0.99 precision, 0.92 recall and  0.95 f1 score.Lastly, we inherit  0.91 precision,  0.90 recall and  0.91 f1-score.

![image](https://github.com/Hem5555/AI-Hand-Gesture-mouse-using-Deep-Learning/assets/121716939/58c47b69-59cb-42da-b28b-37d227d69b44)

![image](https://github.com/Hem5555/AI-Hand-Gesture-mouse-using-Deep-Learning/assets/121716939/65aedb47-846a-4aa3-a0ee-c65a2f1f0320)


Fig 9: Confusion Matrix :Precision,recall  and f1-score


Dataset: https://www.kaggle.com/datasets/hem2023/hand-gesture-dataset










