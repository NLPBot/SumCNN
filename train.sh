#!/bin/sh

python3 FeatureExtractor.py 0
python SumCNN.py
python3 FeatureExtractor.py 1
python predict.py



