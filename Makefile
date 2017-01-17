.PHONY: all run

make:


all:
	(\
		pip install -r ./requirements.txt;\
	)

run:
	sleep 1
	python flickrdownload.py -o pictures
	sleep 1
	python MLS_Hermosa.py $(KNU_DEVICE) &
	sleep 90
	python ws_client.py
	sleep 2
	pkill python
