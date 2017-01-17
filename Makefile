.PHONY: all run

make:
	pip install -r ./requirements.txt

all: ;

run:
	python MLS_Hermosa.py $(KNU_DEVICE) &
	python flickrdownload.py -o pictures &
	sleep 60
	python ws_client.py
