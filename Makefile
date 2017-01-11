.PHONY: all run

all: ;

run:
	python MLS_Hermosa.py $(KNU_DEVICE) &
	sleep 60
	python ws_client.py
