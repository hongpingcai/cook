#!/usr/bin/env python
'''
Title		:step3_test.py
Description	:Input an image (either sushi or sandwich), predict what it is.
Usage		:
    python step3_test.py
    python step3_test.py --i ../img/sandwich/train_4386.jpg
OUTPUT		:img/train.txt
		 img/test.txt
Author		:Hongping Cai
Data		:26/06/2017	
'''

import numpy as np
import os, os.path
import sys, getopt
import caffe
import argparse
import Image
import time

DIR_ROOT = "/media/deepthought/DATA/Hongping/Codes/cook/"
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Predict suchi or sandwich')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--prototxt', dest='prototxt',
                        help='deploy prototxt file defining the network',
                        default=DIR_ROOT + 'prototxt/deploy.prototxt', type=str)
    parser.add_argument('--model', dest='caffemodel',
                        help='caffe model to test',
                        default= DIR_ROOT + 'model/caffenetcook_lr0001_fix3_iter_400.caffemodel', type=str)
    parser.add_argument('--mean', dest='file_mean',
                        help='the mean file (*.npy)',
                        default= DIR_ROOT + 'model/ilsvrc_2012_mean.npy', type=str)
    parser.add_argument('--i', dest='file_input',
                        help='input image (either suchi or sandwich',
                        default='../img/sushi/train_8886.jpg', type=str)


    #if len(sys.argv) == 1:
    #   parser.print_help()
    #   sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)
    
    # pdb.set_trace();
    if not os.path.exists(args.prototxt):
        raise IOError(('{:s} not found.\n').format(args.prototxt))

    if not os.path.exists(args.caffemodel):
        raise IOError(('{:s} not found.\nDid you run sh step2_train.sh\n').format(args.caffemodel))

    if not os.path.exists(args.file_input):
        raise IOError(('{:s} not found.\n').format(args.file_input))
   

    file_input_, file_input_ext = os.path.splitext(args.file_input)
    if file_input_ext == ".txt": # a list of image files with ground-truth labels (either 0,1)
   	lines = tuple(open(args.file_input, 'r'))
   	im_files = []
	im_labels= []
	for line in lines:
		words = line.split(" ")
		im_files.append(words[0])
		im_labels.append(int(words[1]))
		
    elif file_input_ext == "":#all jpg images in the folder
	n_members = len([name for name in os.listdir(file_input_) if name.endswith(".jpg")])
	print 'There are',n_members,'jpg images in',file_input_
        im_files = []#"" for i in range(n_members)]
	for file in os.listdir(file_input_):
		if file.endswith(".jpg"):
			im_files.append(os.path.join(file_input_,file))
    	im_labels = []
    else:
    	# Read input image and transform it
    	try:
		im_files = []
		im_labels= []
  		im_files.append(args.file_input)
    		im = Image.open(args.file_input)
    		print im.format, im.size, im.mode
		im.show()
    	except IOError:
		print "Cannot open ", args.file_input   
    
    # Load caffe model
    if args.gpu_id<0:
	caffe.set_mode_cpu()
    else:
	caffe.set_mode_gpu()
    	caffe.set_device(args.gpu_id)
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)

    
    #Define image transformers	
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load(args.file_mean).mean(1).mean(1))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)

    net.blobs['data'].reshape(1,3,227,227)

    out_labels = []
    out_probs  = []
    start_time = time.time()
    for im_file in im_files:
    	im = caffe.io.load_image(im_file)#(args.file_input)
    	#im = caffe.io.resize_image(im, [IMAGE_HEIGHT,IMAGE_WIDTH])
    	#print net.blobs['data'].data.shape
    	#print net.blobs['data'].data.shape[2:]
    	#crop_dims = np.array(net.blobs['data'].data.shape[2:])
   	#input_ = caffe.io.oversample(im, crop_dims)# [CROP_HEIGHT,CROP_WIDTH])
    	net.blobs['data'].data[...] = transformer.preprocess('data', im)#input_)    
    	out = net.forward()
    	out_label = out['prob'].argmax()
        out_labels.append(out_label)
        out_probs.append(out['prob'][0,out_label])
    elapsed_time = time.time() - start_time
    
    print "********** Prediction:"
    objs_name = {0:"--*Sandwich*--",1:"--*Sushi*--"}
    for i in range(len(im_files)):
	#print out_labels[i]
	#print out_probs[i]
	print im_files[i],objs_name[out_labels[i]],"(probability:",out_probs[i],")"

    if len(im_labels)>0: #gt labels offered
	accuracy = 0.0
        n_correct = 0
	for i in range(len(out_labels)):
		if out_labels[i]==im_labels[i]:
			n_correct += 1
    	accuracy = n_correct/float(len(out_labels))
	print("Accuracy: {:.3f}".format(accuracy))
    if args.gpu_id<0:
     	print "**********",len(im_files),"images, %s seconds without GPU." % elapsed_time
    else:	
     	print "**********",len(im_files),"images, %s seconds with GPU." % elapsed_time



'''
DIR_ROOT = "/media/deepthought/DATA/Hongping/Codes/cook/"
dir_img  = DIR_ROOT + "img/"
classes = {'sandwich':0, 'sushi':1}

# 

file_input_im = str(sys.argv

# 
file_prototxt = DIR_ROOT + 'prototxt/deploy.prototxt'
file_model    = DIR_ROOT + 'model/caffenetcook_lr001_fix3_iter_400.caffemodel'
net = caffe.Net(file_prototxt, file_model,caffe.TEST)

# Transform the image
'''

