install:
	py -m pip install -r requirements.txt

preprocess:
	py src/preprocess.py

train:
	py src/train.py

all: preprocess train
