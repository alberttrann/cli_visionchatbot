# Multimodal Chatbot with Model Switching

A Python-based chatbot supporting **text and image inputs** with dynamic switching between four models:  
- **Qwen2.5-VL 3B** (multimodal, 4-bit)  
- **Qwen2.5-VL 7B** (multimodal, 4-bit)  
- **Gemma 3 4B** (multimodal, 4-bit)  
- **SmolDocling 256M** (multimodal, full quant)  

---

## Features  
✅ **Dynamic Model Switching**: Change models mid-chat without restarting with a simple "switch" in a new line  
✅ **Multimodal Support**: All models can smoothly handle both reasoning, structured outputs and image processing  
✅ **Memory Optimization**: Explicit GPU management for stability.  
✅ **Quantization Handling**: Support for 4-bit models.  

---

## Requirements  
- Python 3.10+  
- GPU with CUDA support
- CUDA Toolkit  
- Install dependencies:  
  ```bash
  pip install bitsandbytes pillow accelerate
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # Match your CUDA version. Replace cu118 with your setup's version if needed.
  pip install git+https://github.com/huggingface/transformers.git #This is essential for the Gemma model to work 
  ```

---

## Run the script 
- Create and activate virtual environment: python -m venv .venv and .\.venv\Scripts\activate
- run the script: python qwen2.5.py
