# GoogLeNet Image Recognition Demo

## Description
This demonstration loads a pre-trained convolutional neural network based off of ImageNet's 1000
categories (http://image-net.org/challenges/LSVRC/2014/browse-synsets).

## Code Overview
* MLS_Hermosa.py starts a server which feeds images of size 256 x 256 to the Hermosa for Processing
* flickrdownload.py pulls images from a pre-defined flickr repository
* ws_client.py pulls and processes all the images from the folder 'pictures' and outputs the results to output_results.csv

## Operation
* Flickr Login:
  * Username: KNUPATH@yahoo.com
	* Password: KNUP@TH1
* Login to the Flickr account above to upload photos (default photos exist)
	* https://www.flickr.com

## How to run
* Locally:
	* Run 'make all run'
* KWS:
	* NOTE: KWS is only able to run one instance of GoogLeNet at one time
	* Login to KWS: https://sandbox.knuedge.com/Queue
	* Add a new repository with the following URL: https://github.com/knupath/GoogLeNet_Demo.git
	* Run Repository
