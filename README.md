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
	  * Username: support@knupath.com
		* Password: KNUPATH
	* Login to the Flickr account above to upload photos (default photos exist)
		* https://www.flickr.com

## How to run
	* Locally:
		* Run 'make all run'
	* KWS:
		* Login to KWS: https://sandbox.knuedge.com/Queue
		* Add a new repository with the following URL: https://github.com/knupath/GoogLeNet_Demo.git
		* Run Repository
