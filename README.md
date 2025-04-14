# Lightwright Command-Line Chatbot with Model Switching

A Python-based chatbot supporting **text and image inputs** with dynamic switching between four models that are lightweight enough to run with less than 8gb of VRAM, but still punch above its weight in multimodal capabilities
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
✅ **Memory efficiency**: Only 1 model is in the memory at a time. One in 3 remaining model is only loaded when it is called upon. When the next model is loaded, the previous model is automatically unloaded

✅ **Logging**: Has logging for necessary information
```
        print(f"\n🕒 Time: {elapsed:.2f}s | Tokens: {tokens} | ⚡ Tokens/sec: {tps:.2f}")
        print(f"   Input Length: {input_len} | Output Length: {len(decoded_ids)}")
        print(f"   Model Type: {current_model_type} | Model Name: {current_model_name}")
        print(f"   Device: {target_device} | Model Device: {current_model.device}")
```
✅ **Simple navigation**: Pick models with a simple menu-like interface, type switch in a new line to switch model and type exit for exiting the program. 

✅ **Support multi-line input**: You can paste in code snippets with multiple lines, freely type ENTER to write long paragraphs with structured layout as input for the chatbot. 

✅ **Image path detection**: Automatically detects image path, so you can input image as simple as print out the content of all 6 questions in here "C:\Users\alberttran\Pictures\Screenshots\Screenshot 2025-04-06 223447.png"

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

## Demo 
```python
(.venv) PS F:\qwen2.5vl> python qwen2.5.py
Using device: cuda
Select a model:
1 - Qwen2.5-VL 3B
2 - Qwen2.5-VL 7B
3 - Gemma 3 4B
4 - SmolDocling 256M
Your choice: 3 

🔄 Loading model: Gemma 3 4B ...
 -> Using Gemma-specific loading for Gemma 3 4B
Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:35<00:00, 17.99s/it]
✅ Gemma 3 4B loaded.

   Model intended device: cuda:0
=== Chatbot (Gemma 3 4B) ===
Type 'exit' to quit, 'switch' to change models. Type 'END' to submit multi-line input.


[Gemma 3 4B] Enter your prompt:
print out the content of all 6 questions here "C:\Users\alberttran\Pictures\Screenshots\Screenshot 2025-04-06 223447.png"
END 
 -> Preparing input using apply_chat_template for Gemma 3 4B...
   Found image: C:\Users\alberttran\Pictures\Screenshots\Screenshot 2025-04-06 223447.png
   Moving inputs to target device: cuda:0
⏳ Generating response...

💬 Response:
Okay, here's the content of all 6 questions from the image, formatted as text:

**2. Find y'**

a.  y = x² - √x * x + 1 / x + 2

b.  y = √x + √√x

c.  y = x² / (x + 1)

d.  y = x * √(x + 2)

e.  y = ln(x² + 1) - 1 / x

f.  y = eˣ sin(2x + 1)

🕒 Time: 229.78s | Tokens: 127 | ⚡ Tokens/sec: 0.55

[Gemma 3 4B] Enter your prompt:
switch

Available models:
1 - Qwen2.5-VL 3B
2 - Qwen2.5-VL 7B
3 - Gemma 3 4B
4 - SmolDocling 256M
Select model number to load: 4
Unloading previous model: Gemma 3 4B...
Previous model unloaded.

🔄 Loading model: SmolDocling 256M ...
 -> Using original loading logic for SmolDocling 256M
✅ SmolDocling 256M loaded.

   Model intended device: cuda:0
--------------------
=== Chatbot (SmolDocling 256M) ===
Type 'exit' to quit, 'switch' to change models. Type 'END' to submit multi-line input.


[SmolDocling 256M] Enter your prompt:
print out the content of all 6 questions in here "C:\Users\alberttran\Pictures\Screenshots\Screenshot 2025-04-06 223447.png"
END 
 -> Preparing input using apply_chat_template for SmolDocling 256M...
   Found image: C:\Users\alberttran\Pictures\Screenshots\Screenshot 2025-04-06 223447.png
Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation=True` to explicitly truncate examples to max length. Defaulting to 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy more precisely by providing a specific strategy to `truncation`.
   Moving inputs to target device: cuda
⏳ Generating response...

💬 Response:
<formula> <loc_ 0> <loc_ 0> <loc_ 500> <loc_ 500>\begin{array} { c } 2 . \ F i n d \ y ^ { \prime } \\ \\ a . \ y = x ^ { 2 } - x \sqrt { x } + \frac { 1 } { x } + 2 \\ \\ d . \ y = x \sqrt { x } + 2 \end{array} \begin{array} { l } b . \ y = \sqrt { x } + \sqrt { x } \\ \\ e . \ y = \ln ( x ^ { 2 } + 1 ) - \frac { 1 } { x } \\ \\ f . \ y = e ^ { x } \sin ( 2 x + 1 ) \\ \\ \end{array} </formula>     

🕒 Time: 50.74s | Tokens: 179 | ⚡ Tokens/sec: 3.53

[SmolDocling 256M] Enter your prompt:
switch

Available models:
1 - Qwen2.5-VL 3B
2 - Qwen2.5-VL 7B
3 - Gemma 3 4B
4 - SmolDocling 256M
Select model number to load: 1
Unloading previous model: SmolDocling 256M...
Previous model unloaded.

🔄 Loading model: Qwen2.5-VL 3B ...
 -> Using original loading logic for Qwen2.5-VL 3B
✅ Qwen2.5-VL 3B loaded.

   Model intended device: cuda:0
--------------------
=== Chatbot (Qwen2.5-VL 3B) ===
Type 'exit' to quit, 'switch' to change models. Type 'END' to submit multi-line input.


[Qwen2.5-VL 3B] Enter your prompt:
print out the content of all 6 questions in here "C:\Users\alberttran\Pictures\Screenshots\Screenshot 2025-04-06 223447.png"
END 
 -> Preparing input using apply_chat_template for Qwen2.5-VL 3B...
   Found image: C:\Users\alberttran\Pictures\Screenshots\Screenshot 2025-04-06 223447.png
   Moving inputs to target device: cuda
⏳ Generating response...

💬 Response:
Sure, here are the derivatives for each question:

2. Find \( y' \)

a. \( y = x^2 - x\sqrt{x} + \frac{1}{x} + 2 \)

To find the derivative, we use the power rule and the chain rule:
\[ y' = \frac{d}{dx}(x^2) - \frac{d}{dx}(x\sqrt{x}) + \frac{d}{dx}\left(\frac{1}{x}\right) + \frac{d}{dx}(2) \]
\[ y' = 2x - (1 + \frac{1}{2}x^{-\frac{1}{2}}) - \frac{1}{x^2} + 0 \]
\[ y' = 2x - 1 - \frac{1}{2x^{3/2}} - \frac{1}{x^2} \]

b. \( y = \sqrt{x} + \sqrt{x} \)

This can be simplified to:
\[ y = 2\sqrt{x} \]
The derivative is:
\[ y' = \frac{d}{dx}(2x^{1/2}) = 2 \cdot \frac{1}{2} x^{-1/2} = \frac{1}{\sqrt{x}} \]

c. \( y = \frac{x^2}{x+1} \)

Using the quotient rule:
\[ y' = \frac{(x+1)\frac{d}{dx}(x^2) - x^2\frac{d}{dx}(x+1)}{(x+1)^2} = \frac{(x+1)2x - x^2(1)}{(x+1)^2} = \frac{2x^2 + 2x - x^2}{(x+1)^2} = \frac{x^2 + 2x}{(x+1)^2} \]

d. \( y = x\sqrt{x+2} \)

Using the product rule:
\[ y' = x\frac{d}{dx}(\sqrt{x+2}) + (\sqrt{x+2})\frac{d}{dx}(x) = x\cdot \frac{1}{2\sqrt{x+2}} + \sqrt{x+2} = \frac{x}{2\sqrt{x+2}} + \sqrt{x+2} \]

e. \( y = \ln(x^2 + 1) - \frac{1}{x} \)

Using the chain rule:
\[ y' = \frac{d}{dx}[\ln(x^2 + 1)] - \frac{d}{dx}\left(\frac{1}{x}\right) = \frac{2x}{x^2 + 1} + \frac{1}{x^2} \]

f. \( y = e^x \sin(2x + 1) \)

Using the product rule:
\[ y' = e^x \cos(2x + 1) \cdot 2 + e^x \sin(2x + 1) = 2e^x \cos(2x + 1) + e^x \sin(2x + 1) \]

So the final answers are:

a. \( y' = 2x - 1 - \frac{1}{2x^{3/2}} - \frac{1}{x^2} \)
b. \( y' = \frac{1}{\sqrt{x}} \)
c. \( y' = \frac{x^2 + 2x}{(x+1)^2} \)
d. \( y' = \frac{x}{2\sqrt{x+2}} + \sqrt{x+2} \)
e. \( y' = \frac{2x}{x^2 + 1} + \frac{1}{x^2} \)
f. \( y' = 2e^x \cos(2x + 1) + e^x \sin(2x + 1) \)

🕒 Time: 239.43s | Tokens: 869 | ⚡ Tokens/sec: 3.63
```

```python
(.venv) PS F:\qwen2.5vl> python qwen2.5.py
Using device: cuda
Select a model:
1 - Qwen2.5-VL 3B
2 - Qwen2.5-VL 7B
3 - Gemma 3 4B
4 - SmolDocling 256M
Your choice: 3

🔄 Loading model: Gemma 3 4B ...
 -> Using Gemma-specific loading for Gemma 3 4B
Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:15<00:00,  7.57s/it]
✅ Gemma 3 4B loaded.

   Model parameter device detected: cuda:0
=== Chatbot (Gemma 3 4B) ===
Type 'exit' to quit, 'switch' to change models. Type 'END' to submit multi-line input.


[Gemma 3 4B] Enter your prompt:
hi
END 
 -> Preparing input using apply_chat_template for Gemma 3 4B...
   Moving inputs to target device: cuda
⏳ Generating response...

💬 Response:
Hi there! How’s it going?

Is there anything you’d like to talk about, or were you just saying hello? 😊

Do you want to:

*   **Chat about something specific?** (e.g., a hobby, a question you have, a story you want to tell)
*   **Play a game?** (e.g., 20 questions, a simple word game)
*   **Get some information?** (e.g., a fact, a definition, a summary of something)

🕒 Time: 15.69s | Tokens: 117 | ⚡ Tokens/sec: 7.46

[Gemma 3 4B] Enter your prompt:
switch

Available models:
1 - Qwen2.5-VL 3B
2 - Qwen2.5-VL 7B
3 - Gemma 3 4B
4 - SmolDocling 256M
Select model number to load: 1
Unloading previous model: Gemma 3 4B...
Previous model unloaded.

🔄 Loading model: Qwen2.5-VL 3B ...
 -> Using original loading logic for Qwen2.5-VL 3B
✅ Qwen2.5-VL 3B loaded.

   Model parameter device detected: cuda:0
--------------------
=== Chatbot (Qwen2.5-VL 3B) ===
Type 'exit' to quit, 'switch' to change models. Type 'END' to submit multi-line input.


[Qwen2.5-VL 3B] Enter your prompt:
hi
END 
 -> Preparing input using apply_chat_template for Qwen2.5-VL 3B...
   Moving inputs to target device: cuda
⏳ Generating response...

💬 Response:
Hello! How can I assist you today?

🕒 Time: 1.14s | Tokens: 10 | ⚡ Tokens/sec: 8.80

[Qwen2.5-VL 3B] Enter your prompt:
switch

Available models:
1 - Qwen2.5-VL 3B
2 - Qwen2.5-VL 7B
3 - Gemma 3 4B
4 - SmolDocling 256M
Select model number to load: 3
Unloading previous model: Qwen2.5-VL 3B...
Previous model unloaded.

🔄 Loading model: Gemma 3 4B ...
 -> Using Gemma-specific loading for Gemma 3 4B
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:18<00:00,  9.34s/it]
✅ Gemma 3 4B loaded.

   Model parameter device detected: cuda:0
--------------------
=== Chatbot (Gemma 3 4B) ===
Type 'exit' to quit, 'switch' to change models. Type 'END' to submit multi-line input.


[Gemma 3 4B] Enter your prompt:
yo 
END 
 -> Preparing input using apply_chat_template for Gemma 3 4B...
   Moving inputs to target device: cuda
⏳ Generating response...

💬 Response:
Hey there! How's it going? Is there anything you’d like to chat about, or were you just saying hi? 😊

Let me know if you want to:

*   Talk about something specific
*   Play a game
*   Get some information
*   Just have a conversation

🕒 Time: 26.22s | Tokens: 65 | ⚡ Tokens/sec: 2.48

[Gemma 3 4B] Enter your prompt:
switch

Available models:
1 - Qwen2.5-VL 3B
2 - Qwen2.5-VL 7B
3 - Gemma 3 4B
4 - SmolDocling 256M
Select model number to load: 4
Unloading previous model: Gemma 3 4B...
Previous model unloaded.

🔄 Loading model: SmolDocling 256M ...
 -> Using original loading logic for SmolDocling 256M
✅ SmolDocling 256M loaded.

   Model parameter device detected: cuda:0
--------------------
=== Chatbot (SmolDocling 256M) ===
Type 'exit' to quit, 'switch' to change models. Type 'END' to submit multi-line input.


[SmolDocling 256M] Enter your prompt:
hi how are you
END 
 -> Preparing input using apply_chat_template for SmolDocling 256M...
   Moving inputs to target device: cuda
⏳ Generating response...

💬 Response:
Hi, how are you?

🕒 Time: 0.83s | Tokens: 7 | ⚡ Tokens/sec: 8.42

[SmolDocling 256M] Enter your prompt:
switch

Available models:
1 - Qwen2.5-VL 3B
2 - Qwen2.5-VL 7B
3 - Gemma 3 4B
4 - SmolDocling 256M
Select model number to load: 3
Unloading previous model: SmolDocling 256M...
Previous model unloaded.

🔄 Loading model: Gemma 3 4B ...
 -> Using Gemma-specific loading for Gemma 3 4B
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:17<00:00,  8.59s/it]
✅ Gemma 3 4B loaded.

   Model parameter device detected: cuda:0
--------------------
=== Chatbot (Gemma 3 4B) ===
Type 'exit' to quit, 'switch' to change models. Type 'END' to submit multi-line input.


[Gemma 3 4B] Enter your prompt:
a 20-line poem about quantum physics in anime style please
END 
 -> Preparing input using apply_chat_template for Gemma 3 4B...
   Moving inputs to target device: cuda
⏳ Generating response...

💬 Response:
Okay, here's a 20-line poem about quantum physics, styled with an anime feel – aiming for a blend of wonder, mystery, and a touch of dramatic flair:

**(Image: Start with a swirling, iridescent nebula background, maybe a stylized silhouette of a lone figure gazing upwards)**

1.  The world, they say, is solid, bright, and true –
2.  But deeper still, a secret waits for you!
3.  Quantum whispers, a shimmering haze,
4.  Where particles dance in a dizzying maze.

5.  (Sparkle effect) Uncertainty’s a hero bold,
6.  A wave, a particle, a story untold!
7.  Schrödinger’s cat, a paradox profound,
8.  Existing both alive and lost, unbound.

9.  (Dramatic zoom-in on a single electron)
10. Observation’s touch, a magic bright,
11. Collapsing waves with a focused light!
12. Superposition, a thrilling plea,
13. Multiple selves, for all to see (almost!).

14. Entangled twins, across the vast unknown,
15. Linked by fate, a connection shown.
16. (Fast-motion sequence)  Particles leap and flow,
17.  Breaking the rules, a vibrant glow!

18.  The universe hums, a quantum song,
19.  Mysteries unfold, where we belong.
20. (Fade to a single, glowing particle)  “To understand… is to begin…”


---

**Notes & Style Considerations:**

*   **Keywords:** I’ve incorporated key terms like “uncertainty,” “superposition,” “entanglement,” “Schrödinger’s cat,” and “wave-particle duality.”
*   **Anime Flair:**  I've used phrases like “shimmering haze,” “dizzying maze,” “magic bright,” “vibrant glow,” “fast-motion sequence,” and “sparkle effect” to evoke an anime aesthetic.  The descriptions are a little more dramatic and stylized.
*   **Visuals:**  Imagine this poem accompanied by dynamic visuals—swirling colors, particle trails, glowing effects, and perhaps a lonely, thoughtful protagonist.
*   **Tone:**  A blend of awe, mystery, and a touch of the slightly unsettling (as quantum physics can be!).

To help me tailor this even more, could you tell me:

*   Is there a specific anime style you’d like me to lean into (e.g., Studio Ghibli, action-oriented, magical girl)?
*   Are there any particular concepts within quantum physics you’d like me to emphasize?

🕒 Time: 77.42s | Tokens: 586 | ⚡ Tokens/sec: 7.57

[Gemma 3 4B] Enter your prompt:
exit 
👋 Exiting chatbot.

Cleaning up GPU memory...
Cleanup complete.
```

## Code to run one model (Qwen2.5-VL-3B)
```python
import re
import os
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
import torch

# Load the 3B, 4-bit quantized Qwen2.5-VL model
print("Loading model... (This may take ~1 minute on first run)")
processor = AutoProcessor.from_pretrained("unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit")
model = AutoModelForImageTextToText.from_pretrained("unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit")
model.to("cuda")  # Move model to GPU
print("Model loaded ✅\n\n=== Qwen 2.5 VL Chatbot (3B) ===")
print("Type 'exit' any time (even mid-input) to quit.")
print("Type 'END' on a new line to submit your multi-line prompt.")

while True:
    print("\nEnter your question (multi-line). Press ENTER for new lines. Type 'END' on a new line to submit:")

    # Manually collect multi-line input
    user_input_lines = []
    while True:
        line = input()

        # If the user wants to exit right away
        if line.strip().lower() == "exit":
            print("Exiting chatbot. Goodbye!")
            exit()  # Immediate exit

        # 'END' signals the end of the multi-line input
        if line.strip().upper() == "END":
            break

        user_input_lines.append(line)

    user_input = "\n".join(user_input_lines).strip()

    # If the user typed nothing (or 'exit' was typed above), just continue
    if not user_input:
        continue

    # If the entire multi-line input is "exit" (somehow), handle it here
    if user_input.lower() == "exit":
        print("Exiting chatbot. Goodbye!")
        break

    # 1. Extract last valid image path from user input (if any)
    matches = re.findall(r'"(.*?)"', user_input)
    image_path = None
    if matches:
        # Take the last match that actually exists on disk
        for candidate in reversed(matches):
            if os.path.exists(candidate):
                image_path = candidate
                break

    # 2. Remove the image path from user input (so text is clean)
    user_question = user_input
    if image_path:
        user_question = user_input.replace(f'"{image_path}"', "").strip()

    # 3. Prepare input messages for the model
    messages = [{"role": "user", "content": []}]
    if image_path:
        messages[0]["content"].append({"type": "image", "image": image_path})
    messages[0]["content"].append({"type": "text", "text": user_question})

    # 4. Apply chat formatting
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True).strip()

    # 5. Process image input
    image_inputs, video_inputs = process_vision_info(messages)

    # 6. Tokenize input
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda")

    # 7. Run inference
    print("\n⏳ Generating response...")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=8192, do_sample=True, temperature=0.9, top_p=0.95, top_k=50, num_beams=1, num_return_sequences=1)

    # 8. Trim unnecessary system messages
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0]

    # 9. Remove system messages if present
    unwanted_text = "system\nYou are a helpful assistant.\nuser\n\nassistant\n"
    if unwanted_text in output_text:
        output_text = output_text.replace(unwanted_text, "").strip()

    # 10. Print final result
    print("\n💬 Qwen:", output_text)
```
