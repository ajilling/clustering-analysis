#!/bin/sh
echo "Begin Testing..."
python3 train.py .000001 .0001 1500 2
python3 train.py .000001 .001 1500 2
python3 train.py .000001 .01 1500 2
python3 train.py .000001 .1 1500 2
python3 train.py .000001 1 1500 2

python3 train.py .00001 .0001 1500 2
python3 train.py .00001 .001 1500 2
python3 train.py .00001 .01 1500 2
python3 train.py .00001 .1 1500 2
python3 train.py .00001 1 1500 2

python3 train.py .0001 .0001 1500 2
python3 train.py .0001 .001 1500 2
python3 train.py .0001 .01 1500 2
python3 train.py .0001 .1 1500 2
python3 train.py .0001 1 1500 2

python3 train.py .001 .0001 1500 2
python3 train.py .001 .001 1500 2
python3 train.py .001 .01 1500 2
python3 train.py .001 .1 1500 2
python3 train.py .001 1 1500 2

python3 train.py .01 .0001 1500 2
python3 train.py .01 .001 1500 2
python3 train.py .01 .01 1500 2
python3 train.py .01 .1 1500 2
python3 train.py .01 1 1500 2

python3 train.py .1 .0001 1500 2
python3 train.py .1 .001 1500 2
python3 train.py .1 .01 1500 2
python3 train.py .1 .1 1500 2
python3 train.py .1 1 1500 2

python3 train.py 1 .0001 1500 2
python3 train.py 1 .001 1500 2
python3 train.py 1 .01 1500 2
python3 train.py 1 .1 1500 2
python3 train.py 1 1 1500 2

python3 train.py 10 .0001 1500 2
python3 train.py 10 .001 1500 2
python3 train.py 10 .01 1500 2
python3 train.py 10 .1 1500 2
python3 train.py 10 1 1500 2
