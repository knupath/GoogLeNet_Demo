.PHONY: all run

make:


all:
	(\
		pip install -r ./requirements.txt;\
	)

run:
	hard-reset -t $(KNU_DEVICE)
	hard-release -t $(KNU_DEVICE)
	python MLS_Hermosa.py $(KNU_DEVICE) &
	python flickrdownload.py -o pictures &
	sleep 80
	python ws_client.py
