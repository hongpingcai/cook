
## Prediction: Sushi or Sandwich

Given an image, this project is to predict it is sushi or sandwich. 

## Quick started
#NOTE: please change your current directory to cook/codes/ before running the code.
Suppose you have cloned the repository into COOK_DIR
'''
cd COOK_DIR/codes
python step3_test.py --i ../img/sandwich/train_1009.jpg 
'''
After a while, you should see the image displayed and its predicted label. stepp3_test.py has more inputs supported, please referred to the 'How to run the codes' section. 
You would see such output on the screen:

********** Prediction:
../img/sandwich/train_1009.jpg --*Sandwich*-- (probability: 0.999996 )
********** 1 images,0.0816650390625 seconds

## Dependencies
1. caffe, pycaffe
See: Installation of caffe (http://caffe.berkeleyvision.org/installation.html)
2. python
See: Downloading python (https://www.python.org/downloads/)

## How to use the codes
NOTE: If you only want to do prediction for one image or a bundle of images, then only need to run step3, as I have uploaded the splitted data list (step1) and the trained model (step2). 
1. step1: randomly split the image set into training set (5/6) and test set (1/6)
'''
cd COOK_DIR/codes
python step1_data_split.py
'''

2. step2: training the caffe model 
Please change the COOK_DIR in step2_train.sh before running.
Also change a few directories in prototxt/train_val.prototxt and prototxt/solver_lr0001_fix3.prototxt.
'''
cd CAFFE_ROOT
sh COOK_DIR/codes/step2_train.sh
'''
After training, you'd better check the learning curve. Please change the caffe_path inside plot_learning_curve.py before running the following codes.  (The output figure is saved as ../log/caffenetcook_learning_curve.png)
'''
cd COOK_DIR/codes
python plot_learning_curve.py ../log/caffenetcook_lr0001_fix3.log ../log/caffenetcook_lr0001_fix3_learning_curve.png
'''

3. step3: prediction with the trained model
Three inputs are surported to predict labels:

a) a single image input
'''
cd COOK_DIR/codes
python step3_test.py --i ../img/sandwich/train_1009.jpg 
'''

b) a directory input. All the jpg images inside this directory will be tested
'''
cd COOK_DIR/codes
python step3_test.py --i ../img/sandwich/
'''

c) a text file (.txt) with images and labels listed on each row. 
'''
cd COOK_DIR/codes
python step3_test.py --i ../img/test.txt
'''
As there is ground-truth labels available, the accuracy will be also outputed at the end.

To note: As default, the prediction uses the model trained from training set. But if you predict images outside of the current dataset, it is better to use the following code to utilize the model trained from the whole dataset (training+test):
'''
cd COOK_DIR/codes
python step3_test.py --i ../img/test.txt --model ../model/alldata_caffenetcook_lr0001_fix3_iter_400.caffemodel
''' 

## Settings 
I have randomly split the given image set into training set (5/6 of the whole set, 670 images, listed in img/train.txt) and test set (1/6 of the whole set, 134 images, listed in img/test.txt). I trained the caffenet model on the two-class classificatin problem by finetuning the caffenet model (pre-trained on ImageNet dataset). I have tried a few parameters, found the following parameters are good for converge and producing high accuracy: 
- fix layers: 3
- base learning rate: 0.0001
- stepsize: 500
- momentum: 0.9
- weight decay; 0.0005

## Results
The following is the learning curve: 

![Alt text](log/caffenetcook_lr0001_fix3_learning_curve.png?raw=true "Title")

After about 200 iterations, the loss on both training set and test set keep stable.

I deploy the trained model from step2 to predict the labels for all the 134 test images. The accuracy (= #number_correct_predict/134) is 86.6%.

Currently on my computer, the training takes about 4 minutes (for 400 iterations) with GPU (Titan X Pacal), the prediction time is about 0.03 seconds/image with GPU, 0.16 seconds/image with CPU. 

## Analysis
To note, I used a very basic caffe model to do the test, considering the small size of the dataset, less GPU memory used, less time trained. If the more advanced models, e.g., VGG-16, GoogleNet, ResNet, are used, the accuracy would usually go up. And also if more training images available, the accuracy would also increase.

The model could be easily modified to predict more labels (e.g., all types of food), as long as enough labeled training images available. 

## Authors
Hongping Cai, hongping.cai.80@gmail.com

## License
If commercial usage of this code, please contact the author first.

