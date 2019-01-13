#!/usr/bin/env bash

chmod 600 ~/.kaggle/kaggle.json
kaggle competitions download -c humpback-whale-identification -p data/
rm -rf data/train
rm -rf data/test
mkdir data/train
mkdir data/test
unzip data/train.zip -d data/train/
unzip data/test.zip -d data/test/
rm data/train.zip
rm data/test.zip
