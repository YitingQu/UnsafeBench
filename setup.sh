export LD_LIBRARY_PATH=""

eval "$(conda shell.bash hook)"
conda create -n llava python=3.10 -y

eval "$(conda shell.bash hook)"
conda activate llava

pip install --upgrade pip setuptools
pip install -e .
pip install \
  torch==2.0.1+cu118 \
  torchvision==0.15.2+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118

pip install flash-attn==1.0.4
pip install -r requirements.txt