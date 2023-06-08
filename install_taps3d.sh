# conda create -n get3d python=3.8
# conda activate get3d
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install ninja xatlas gdown
pip install git+https://github.com/NVlabs/nvdiffrast/
pip install meshzoo ipdb imageio gputil h5py point-cloud-utils imageio imageio-ffmpeg==0.4.4 pyspng==0.1.0
pip install urllib3
pip install scipy tensorboard
pip install click
pip install tqdm
pip install opencv-python==4.5.4.58
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install --upgrade "jax[cpu]"
pip install flax==0.5.3
pip install dm_pix
apt-get install libegl1-mesa-dev
cd nvdiffrast
pip install .
cd ..

