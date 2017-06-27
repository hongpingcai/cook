'''
Title		:step1_data_split.py
Description	:Randomly split the images into training set (5/6) and test set (1/6)
Usage		:python step1_data_split.py
OUTPUT		:img/train.txt, img/train_shuffle.txt
		 img/test.txt,  img/test_shuffle.txt
Author		:Hongping Cai
Data		:26/06/2017	
'''

import numpy as np
import os, os.path

dir_root = "../"
dir_img  = dir_root + "img/"
classes = {'sandwich':0, 'sushi':1}
ratio_train = 5.0/6.0

np.random.seed(0)
tr_files = []
te_files = []
tr_labels = []
te_labels = []
for cls in classes:
	cur_label = classes[cls]
	n_members = len([name for name in os.listdir(dir_img+cls) if name.endswith(".jpg")])
	print '**id',cur_label,cls,n_members,'images'

	shuffled_ids = np.random.permutation(range(n_members))
	cur_ids_train = shuffled_ids[:int(round(n_members*ratio_train))]
	cur_ids_test  = shuffled_ids[int(round(n_members*ratio_train)):]
	cur_files = ["" for i in range(n_members)]
	count = 0
	for file in os.listdir(dir_img+cls):
		if file.endswith(".jpg"):
			#print count,os.path.join(dir_img+cls,file)
			cur_files[count] = os.path.join(dir_img+cls,file)
			count = count+1

	for id in cur_ids_train:
		tr_files.append(cur_files[id])
		tr_labels.append(cur_label)
	for id in cur_ids_test:
		te_files.append(cur_files[id])
		te_labels.append(cur_label)
	print 'len(tr_files)',len(tr_files)
	print 'len(tr_labels)',len(tr_labels)
	print 'len(te_files)',len(te_files)
	print 'len(te_labels)',len(te_labels)

#write the training file
txt_file = dir_img + "train.txt"
with open(txt_file,"w") as fid:
	for i in range(len(tr_files)):
		fid.write("%s %d\n" % (tr_files[i], tr_labels[i]))

#write the test file
txt_file = dir_img + "test.txt"
with open(txt_file,"w") as fid:
	for i in range(len(te_files)):
		fid.write("%s %d\n" % (te_files[i], te_labels[i]))

np.random.seed(0)
#write the training file, shuffled
ids_shuffle = np.random.permutation(range(len(tr_files)))

txt_file = dir_img + "train_shuffle.txt"
with open(txt_file,"w") as fid:
	for i in range(len(tr_files)):
		cur_i = ids_shuffle[i]
		fid.write("%s %d\n" % (tr_files[cur_i], tr_labels[cur_i]))

#write the test file, shuffled
ids_shuffle = np.random.permutation(range(len(te_files)))
txt_file = dir_img + "test_shuffle.txt"
with open(txt_file,"w") as fid:
	for i in range(len(te_files)):
		cur_i = ids_shuffle[i]
		fid.write("%s %d\n" % (te_files[cur_i], te_labels[cur_i]))
	
		
	












