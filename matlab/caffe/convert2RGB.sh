#!/usr/bin/env sh

#for synset in `ls -d /home/sqiu/dataset/ILSVRC2012/train_resize_2/n*`;do
#  echo $synset
#  for img in `ls ${synset}/*.JPEG`;do
#    convert $img -colorspace RGB $img
#    echo $img
#  done
#done

synset=/home/sqiu/dataset/ILSVRC2012/val_resize_2
for img in ls ${synset}/*.JPEG;do
  convert $img -colorspace RGB $img
  echo $img
done
