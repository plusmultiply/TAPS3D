#!/bin/bash
IMG_ROOT=$1

cd data
python step_1.py
python step_2.py --render_path ${IMG_ROOT}
python step_3.py --render_path ${IMG_ROOT}
cd ..