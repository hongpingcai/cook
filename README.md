
## Prediction: Sushi or Sandwich

Given an image, this project is to predict it is sushi or sandwich.

## Dependencies
- caffe, pycaffe (http://caffe.berkeleyvision.org/installation.html)

- python (https://www.python.org/downloads/) 

## Quick start
Suppose you have cloned the repository into COOK_DIR

First, you need download the models from (https://drive.google.com/open?id=0BxcbdpkZeozjcnVoTW84SlhwVjg) and extract them into COOK_DIR/model/

Now, run the prediction with an input image:

```
cd COOK_DIR/codes
python step3_test.py --i ../img/your_img.jpg 

```
The codes defaultly use GPU. If you don't have GPU, just add '--gpu -1' at the end of the above command.

## How to use the codes

Before using the codes, you need first download the sushi-and-sandwich dataset (I used for training and test) as follows. 

```
cd COOK_DIR
wget http://research.us-east-1.s3.amazonaws.com/public/sushi_or_sandwich_photos.zip
unzip sushi_or_sandwich_photos.zip
rm sushi_or_sandwich_photos.zip
mv sushi_or_sandwich_photos/* img/.
rm -r sushi_or_sandwich_photos/
```

*NOTE*: If you only want to do prediction for one image or a bundle of images, then only need run step3, as I have uploaded the splitted data list (step1) and the trained model (step2). 

**step1**: Randomly split the image set into training set (5/6) and test set (1/6)
```
cd COOK_DIR/codes
python step1_data_split.py
```

**step2**: Train the caffe model 

Please change the CAFFE_PATH in step2_train.sh before running:
```
cd COOK_DIR/codes
sh step2_train.sh
```
After training, you'd better check the learning curve. Please change the caffe_path in plot_learning_curve.py before running the following codes.  The output figure is saved as COOK_DIR/log/caffenetcook_lr0001_fix3_learning_curve.png
```
cd COOK_DIR/codes
python plot_learning_curve.py ../log/caffenetcook_lr0001_fix3.log ../log/caffenetcook_lr0001_fix3_learning_curve.png
```

**step3**: Prediction with the trained model.

Three inputs are surported to predict labels:

- a single image input
```
cd COOK_DIR/codes
python step3_test.py --i ../img/sandwich/train_1009.jpg 
```

- a directory input. All the jpg images inside this directory will be tested
```
cd COOK_DIR/codes
python step3_test.py --i ../img/sandwich/
```

- a text file (.txt) with images and labels listed on each row. 
```
cd COOK_DIR/codes
python step3_test.py --i ../img/test.txt
```
As there are ground-truth labels available, the accuracy will be also outputed at the end.

*Note*: As default, the prediction uses the model trained from training set. But if you predict images outside of the current dataset, it is better to use the following code to utilize the model trained from the whole dataset (training+test):
```
cd COOK_DIR/codes
python step3_test.py --i ../img/test.txt --model ../model/alldata_caffenetcook_lr0001_fix3_iter_400.caffemodel
``` 

## Settings 
I have randomly split the given image set into training set (5/6 of the whole set, 670 images, listed in img/train.txt) and test set (1/6 of the whole set, 134 images, listed in img/test.txt). I trained the caffenet model on the two-class classificatin problem by finetuning the caffenet model (pre-trained on ImageNet dataset). I have tried a few parameters, found the following parameters are good for converge and producing high accuracy: 
- fix layers: 3
- base learning rate: 0.0001
- stepsize: 500
- momentum: 0.9
- weight decay; 0.0005

## Results
#### Convergence
The following is the learning curve: 

![Alt text](log/caffenetcook_lr0001_fix3_learning_curve.png?raw=true "Title")

After about 200 iterations, the loss on both training set and test set keep stable.

#### Accuracy
The accuracy (= #img_correct_predict/134) on the 134 test images is **86.6%** with the trained model from step2.

#### Time
Currently on my computer, the training takes about 4 minutes (for 400 iterations) with GPU (Titan X Pascal), the prediction time is about 0.03 seconds/image with GPU, or 0.16 seconds/image with CPU. 

## Analysis
To note, I used a very basic caffe model to do the test, considering the small size of the dataset, less GPU memory used, less time for training. If the more advanced models, e.g., VGG-16, GoogleNet, ResNet, are used, the accuracy would usually go up. And also if more training images available, the accuracy would also increase.

The model could be easily modified to predict more labels (e.g., all types of food), as long as enough labeled training images available. 

## Authors
- Hongping Cai, hongping.cai.80@gmail.com

## License
If commercial usage of this code, please contact the author first.

