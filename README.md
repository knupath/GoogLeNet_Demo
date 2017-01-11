# GoogLeNet Image Recognition Demo

## Description
This demonstration loads a pre-trained convolutional neural network based off of ImageNet's 1000
categories (http://image-net.org/challenges/LSVRC/2014/browse-synsets).

## Operation
	* MLS_Hermosa.py starts a server which feeds images of size 256 x 256 to the Hermosa for Processing
	* ws_client.py pulls and processes all the images from the folder 'pictures' and outputs the results to output_results.csv

## How to run
	* Run 'make all run'
