######################################################################################
#  Copyright 2017 KNUPATH
#  All rights reserved.
#  KNUPATH Proprietary and Confidential
#
#  File: flickrdownload.py
#
#  Author: Mark Doyle
#
#  Description: Iterates the Flickr image list and downloads all images
#
######################################################################################

import json
import urllib
import argparse
import flickrapi
import os

######################################################################################
# Flickr helpers
######################################################################################

def do_flickr_search(**kwargs):
	'''
	Error check the call to Flickr
	Display an error and exit
	'''
	results = photos = flickr.photos.search(**kwargs)
	parsed_r = json.loads(results.decode('utf-8'))
	# If the status is NOT OK, display and exit
	if parsed_r['stat'] != u'ok':
		print
		print "ERROR:", parsed_r[u'code'], ': ', parsed_r[u'message']
		print
		exit()
	return parsed_r

#-----------------------------------------------------

def convert_flickr_to_url(json_data):
	'''
	Disassemble the JSON and convert to the URL for accessing an image
	'''
	farm = str(photo_dict["farm"])
	server = str(photo_dict["server"])
	id = str(photo_dict["id"])
	secret = str(photo_dict["secret"])
	url = "https://farm" + farm + ".staticflickr.com/" + server + "/" + id + "_" + secret
	# '_n' selects the size of 320 on the longer edge
	#      see https://www.flickr.com/services/api/misc.urls.html
	url += "_n.jpg"
	return  url

#-----------------------------------------------------

def get_title(json_data):
	'''
	Grab the title of this image
	'''
	return str(photo_dict["title"])

#-----------------------------------------------------

# Blank title is "used" to prevent a blank filemane
used_names = ['']
keepcharacters = ('.','_')

def generate_filename(json_data):
	'''
	Create a filename
	Use the image title unless it has been used
		else use the internal namu used by Flickr
	'''
	name = get_title(json_data)
	if name in used_names:
		name = str(photo_dict["secret"])
	file_name = "".join(c for c in name if c.isalnum() or c in keepcharacters).rstrip()
	# Don't reuse names
	used_names.append(file_name)
	return file_name + ".jpg"


######################################################################################
# Main
######################################################################################

# Setup Args
parser = argparse.ArgumentParser(description='Image download from Flickr')
parser.add_argument('-o','--output_directory',help='Image destination directory',
					default = '')
args = parser.parse_args()

# Need all the info to access flickr
api_key = u'96fb9ada6311c5be0fdd5cc1fab50007'
api_secret = u'7b3389b368278a5a'
user_id = u'146755944@N05'

# Grab the image list from flickr
flickr = flickrapi.FlickrAPI(api_key, api_secret, format='json')
photo_info = do_flickr_search(user_id=user_id)

# If folder doesn't exist, create it
if args.output_directory:
	if not os.path.isdir(args.output_directory):
		os.mkdir(args.output_directory)

# Process the first page of images, default = up to 100
for photo_dict in photo_info["photos"]["photo"]:
	url = convert_flickr_to_url(photo_dict)
	out_filename = os.path.join(args.output_directory, generate_filename(photo_dict))
	print
	print "Retrieving '" + get_title(photo_dict) + "' from:", url
	print "  Saving as:", out_filename
	urllib.urlretrieve (url, out_filename)
print
