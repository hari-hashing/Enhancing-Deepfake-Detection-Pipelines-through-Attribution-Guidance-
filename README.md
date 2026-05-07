# Enhancing-Deepfake-Detection-Pipelines-through-Attribution-Guidance-
This repo represents the way to enhance the deep fake detection pipeline through the use of multiple training time interventions and validation through various attribution based methods   

# Set Up Environment
chmod u+x setup_env.sh
bash setup_env.sh

# Training Code Snippet

python main.py train --config configs/default.yaml

# Testing Code Snippet

python main.py eval --config configs/default.yaml

# Installs the torch_ort and onnxruntime-training Python packages
pip install torch-ort

# Configures onnxruntime-training to work with user's PyTorch installation
python -m torch_ort.configure

