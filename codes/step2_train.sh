#!/usr/bin/env sh


TOOLS=./build/tools
COOK_DIR=/media/deepthought/DATA/Hongping/Codes/cook
MODEL=$COOK_DIR/model
PROTOTXT=$COOK_DIR/prototxt
RECORD=$COOK_DIR/log
 
GLOG_logtostderr=1 $TOOLS/caffe train \
--gpu 0 \
--solver $PROTOTXT/solver_lr0001_fix3.prototxt \
--weights $MODEL/bvlc_reference_caffenet.caffemodel \
2>&1 | tee $RECORD/caffenetcook_lr0001_fix3.log

#After training, run this for plotting the learning curve
#python plot_learning_curve.py ../log/caffenetcook.log ../log/caffenetcook_learning_curve.png

echo "Done."
