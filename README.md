# cli_visionchatbot
A lightweight vision chatbot that works in command line interface, combining 4 open-source models under 7B - Qwen2.5-VL-3B and Qwen2.5-VL-7B, both are from Unsloth and quantized to 4 bits; Gemma 3 4b, is also loaded in 4-bit quantization; SmolDocling, a lightweight model in the SmolVLM family which only has 256M parameters


# Create a virtual environment
python -m venv .venv
# Activate the virtual environment (Windows)
.venv\Scripts\activate
# (Linux/macOS)
# source .venv/bin/activate

# Install essential packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # Replace cu118
pip install transformers
pip install accelerate
pip install Pillow
pip install bitsandbytes  # MAKE SURE THIS CUDA/TORCH
pip install git+https://github.com/huggingface/transformers.git 


# Install extra components in case these fail
pip install --upgrade --force-reinstall protobuf
pip install sentencepiece
pip install safetensors

# Run the script
python qwen2.5.py
