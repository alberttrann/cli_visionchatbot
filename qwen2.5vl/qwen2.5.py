import os
import re
import time
import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    Gemma3ForConditionalGeneration,
    BitsAndBytesConfig,
)
# Import traceback for better error reporting
import traceback
# Import PIL for image loading
from PIL import Image
# Import gc
import gc

# --- Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Updated model_config using official Gemma path and distinct types
model_config = {
    "1": {"name": "Qwen2.5-VL 3B", "path": "unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit", "type": "image_text"},
    "2": {"name": "Qwen2.5-VL 7B", "path": "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit", "type": "image_text"},
    "3": {"name": "Gemma 3 4B", "path": "google/gemma-3-4b-it", "type": "gemma"}, # Official path
    "4": {"name": "SmolDocling 256M", "path": "ds4sd/SmolDocling-256M-preview", "type": "image_text"},
}

# --- Global Variables ---
current_model = None
current_processor = None
current_model_name = ""
current_model_type = ""

# --- Model Loading Function (Branched Logic - Force Gemma device map) ---
def load_model(choice):
    global current_model, current_processor, current_model_name, current_model_type, device # Ensure device is accessible

    # Unload previous model if exists
    if current_model is not None:
        print(f"Unloading previous model: {current_model_name}...")
        # Start clean-up process
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize() # Add synchronization point
        gc.collect() # Run garbage collector again
        if current_model: del current_model
        if current_processor: del current_processor
        current_model = None
        current_processor = None
        print("Previous model unloaded.")

    # Get config for chosen model
    config = model_config.get(choice)
    if not config:
        print("Invalid choice!")
        return False

    print(f"\n🔄 Loading model: {config['name']} ...")
    model_path = config["path"]
    current_model_type = config["type"] # Store model type

    try:
        # Determine recommended dtype for computations/loading non-quantized parts
        model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        if current_model_type == "gemma":
            # --- Gemma Loading (Specific logic - Force device map) ---
            print(f" -> Using Gemma-specific loading for {config['name']}")
            bnb_config_gemma = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=model_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            current_processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            current_model = Gemma3ForConditionalGeneration.from_pretrained(
                model_path,
                quantization_config=bnb_config_gemma,
                torch_dtype=model_dtype,
                # Force mapping to the specific CUDA device instead of "auto"
                device_map=device, # Use the global device variable ("cuda" or "cuda:0")
                trust_remote_code=True
            ).eval()

        else: # --- image_text Models (Qwen, SmolDocling - Original Loading Logic) ---
            print(f" -> Using original loading logic for {config['name']}")
            current_processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
            current_model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=model_dtype,
                trust_remote_code=True
            )
            # Manually move model to device
            current_model.to(device)
            current_model.eval()


        current_model_name = config["name"]
        print(f"✅ {current_model_name} loaded.\n")
        # Check device after loading
        try:
             # Check the device of a parameter, as model.device might be misleading with device_map
             param_device = next(current_model.parameters()).device
             print(f"   Model parameter device detected: {param_device}")
        except Exception as dev_e:
              print(f"   Could not determine model parameter device automatically ({dev_e}).")

        return True

    except Exception as e:
        print(f"❌ Error loading model '{config['name']}':")
        traceback.print_exc()
        current_model = None
        current_processor = None
        current_model_name = ""
        current_model_type = ""
        if torch.cuda.is_available():
             torch.cuda.empty_cache()
        return False

# --- Chat Loop Function (Unified Input Processing using apply_chat_template) ---
def chat_loop(current_choice):
    global current_model_name, current_model_type, current_processor, current_model, device # Ensure device is accessible

    if not current_model or not current_processor:
         print("Model not loaded. Cannot start chat loop.")
         return

    print(f"=== Chatbot ({current_model_name}) ===")
    print("Type 'exit' to quit, 'switch' to change models. Type 'END' to submit multi-line input.\n")

    while True:
        print(f"\n[{current_model_name}] Enter your prompt:")
        lines = []
        line = ""
        while True:
            # ... (inner input loop remains the same - exit, switch, END) ...
            try:
                line = input()
            except EOFError:
                print("Input stream ended. Exiting.")
                return

            line_strip_lower = line.strip().lower()
            if line_strip_lower == "exit":
                print("👋 Exiting chatbot.")
                return
            if line.strip().lower() == "switch":
                # ... (switch logic remains the same, ensures chat_loop restarts) ...
                print("\nAvailable models:")
                for key, conf in model_config.items():
                    print(f"{key} - {conf['name']}")
                new_choice = input("Select model number to load: ").strip()
                if new_choice in model_config:
                     if load_model(new_choice):
                         print("-" * 20) # Separator
                         return chat_loop(new_choice) # Restart loop with new choice
                     else:
                         print("Failed to load new model. Staying with the current model.")
                else:
                     print("Invalid selection. Staying with the current model.")
                lines = [] # Reset lines
                break # Break inner loop to re-prompt

            if line.strip().upper() == "END":
                break
            lines.append(line)

        if not lines: # Handles empty input after switch or END
             continue

        user_input = "\n".join(lines).strip()
        if not user_input:
            continue

        # Initialize variables for this loop iteration
        inputs = None
        input_len = 0
        messages = [] # Common messages list
        content = []  # Common content list

        # --- Unified Input Preparation & Processing ---
        try:
            print(f" -> Preparing input using apply_chat_template for {current_model_name}...")
            image_path = None
            image_pil = None

            # 1. Detect image path
            matches = re.findall(r'"(.*?)"', user_input)
            potential_paths = [m for m in matches if os.path.exists(m)]
            user_input_text_only = user_input # Default

            if potential_paths:
                image_path = potential_paths[-1]
                print(f"   Found image: {image_path}")
                user_input_text_only = user_input.replace(f'"{image_path}"', "").strip()
                try:
                    image_pil = Image.open(image_path).convert("RGB")
                    content.append({"type": "image", "image": image_pil}) # Add PIL object
                except Exception as e:
                    print(f"   ⚠️ Error loading image '{image_path}': {e}")
                    image_pil = None

            # Add text content if present
            if user_input_text_only:
                content.append({"type": "text", "text": user_input_text_only})

            if not content:
                 print("   ⚠️ No text or valid image provided.")
                 continue

            messages = [{"role": "user", "content": content}]

            # Apply Processor using apply_chat_template
            inputs = current_processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True # Ensure dict output
            )

            # Determine target device correctly based on how model was loaded
            target_device = None
            # For Gemma, device_map=device was used, so device is hard coded
            # This also should prevent the issues with memory from showing by allocating the GPU device.
            if current_model_type == 'gemma':
                 target_device = device
            else:
                # For .to(device) models, use the script's device variable
                 target_device = device

            print(f"   Moving inputs to target device: {target_device}")
            # Move Tensors to target device
            processed_inputs = {}
            for k, v in inputs.items():
                if k == "pixel_values":
                    # Use model's compute dtype for images
                    target_dtype = getattr(current_model, 'dtype', torch.float16)
                    processed_inputs[k] = v.to(target_device, dtype=target_dtype)
                else:
                    # Preserve original dtype for other inputs (like input_ids=long)
                    processed_inputs[k] = v.to(target_device)
            inputs = processed_inputs


            # Get input_len
            input_ids_tensor = inputs.get("input_ids", None)
            input_len = input_ids_tensor.shape[-1] if input_ids_tensor is not None else 0


        except Exception as e:
            print(f"❌ Input preparation/processing error:")
            traceback.print_exc()
            continue

        # --- Generation ---
        if inputs is None:
             print("   ⚠️ Skipping generation due to input preparation failure.")
             continue

        print("⏳ Generating response...")
        start = time.time()
        try:
            # Get Tokenizer IDs
            pad_token_id_to_use = None
            eos_token_id_to_use = None
            if hasattr(current_processor, 'tokenizer') and current_processor.tokenizer is not None:
                tokenizer = current_processor.tokenizer
                if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
                    pad_token_id_to_use = tokenizer.pad_token_id
                elif hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
                    pad_token_id_to_use = tokenizer.eos_token_id
                    eos_token_id_to_use = tokenizer.eos_token_id
                if eos_token_id_to_use is None and hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
                    eos_token_id_to_use = tokenizer.eos_token_id

            # Prepare generation arguments
            generation_kwargs = {**inputs} # Unpack the dictionary
            generation_kwargs.update({
                "max_new_tokens": 8192,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 50,
            })
            if pad_token_id_to_use is not None:
                generation_kwargs["pad_token_id"] = pad_token_id_to_use

            # Generate
            with torch.no_grad():
                 output = current_model.generate(**generation_kwargs)

        except Exception as e:
             print(f"❌ Generation error:")
             traceback.print_exc()
             continue

        end = time.time()

        # --- Decoding ---
        if not isinstance(output, torch.Tensor) or output.ndim < 2:
             print("   ⚠️ Unexpected output format from generate.")
             decoded = "[Decoding Error]"
             tokens = 0
        else:
            response_ids = output[0]
            decoded_ids = response_ids
            if input_len > 0 and len(response_ids) > input_len:
                 decoded_ids = response_ids[input_len:]
            elif len(response_ids) <= input_len and input_len > 0:
                 print("   ⚠️ Warning: Generated output is not longer than input. Decoding full output.")
                 decoded_ids = response_ids

            if len(decoded_ids) == 0:
                decoded = ""
            else:
                decoded = current_processor.decode(decoded_ids, skip_special_tokens=True)
            tokens = len(decoded_ids)

        elapsed = end - start
        tps = tokens / elapsed if elapsed > 0 else 0

        print(f"\n💬 Response:\n{decoded.strip()}")
        print(f"\n🕒 Time: {elapsed:.2f}s | Tokens: {tokens} | ⚡ Tokens/sec: {tps:.2f}")
        print(f"   Input Length: {input_len} | Output Length: {len(decoded_ids)}")
        print(f"   Model Type: {current_model_type} | Model Name: {current_model_name}")
        print(f"   Device: {target_device} | Model Device: {current_model.device}")

# --- Main Function ---
def main():
    print("Select a model:")
    for k, v in model_config.items():
        print(f"{k} - {v['name']}")

    initial_load_done = False
    while True:
        choice = input("Your choice: ").strip()
        if choice in model_config:
            if load_model(choice):
                initial_load_done = True
                chat_loop(choice)
                # If chat_loop returns (due to 'exit' or a failed switch inside), break the main loop.
                break
            else:
                print("Initial model load failed. Please try selecting another model.")
        elif choice.lower() == 'exit':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please select a number from the list or type 'exit'.")

    if not initial_load_done:
         print("No model was successfully loaded.")


# --- Script Execution ---
if __name__ == "__main__":
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    try:
        main()
    except KeyboardInterrupt:
        print("\nCaught interrupt, exiting.")
    except Exception as e:
        print("\n❌ An unexpected error occurred in the main execution:")
        traceback.print_exc()
    finally:
        # Clean up GPU memory
        if torch.cuda.is_available():
            print("\nCleaning up GPU memory...")
            if 'current_model' in globals() and current_model: del current_model
            if 'current_processor' in globals() and current_processor: del current_processor
            torch.cuda.empty_cache()
            print("Cleanup complete.")