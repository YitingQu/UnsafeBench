# Set environment variable
export LD_LIBRARY_PATH=""
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

eval "$(conda shell.bash hook)"
conda create -n lavis python=3.10 -y
conda activate lavis
pip install --upgrade pip

# Install packages from requirements.txt (avoiding duplicates and conflicts)
pip install contexttimer
pip install decord
pip install "diffusers<=0.16.0"
pip install "einops>=0.4.1"
pip install fairscale==0.4.4
pip install ftfy
pip install iopath
pip install ipython
pip install omegaconf
pip install opencv-python-headless==4.5.5.64
pip install opendatasets
pip install packaging
pip install pandas
pip install plotly
pip install pre-commit
pip install pycocoevalcap
pip install pycocotools
pip install python-magic
pip install scikit-image
pip install sentencepiece
pip install streamlit
pip install timm==0.4.12
pip install tqdm
pip install transformers==4.37.2
pip install webdataset
pip install wheel
pip install soundfile
pip install nltk
pip install easydict==1.9
pip install h5py
pip install seaborn
pip install open3d==0.16.0
pip install pyyaml_env_tag==0.1

# Install PyTorch with CUDA support early
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121

pip install --upgrade spacy thinc pydantic
pip install moviepy==1.0.3 imageio==2.33.0 imageio_ffmpeg==0.4.2
pip install peft==0.5.0
# Install numpy first to avoid compatibility issues
pip install numpy==1.26.4
pip install datasets
pip install huggingface_hub

echo "Setup completed successfully!"