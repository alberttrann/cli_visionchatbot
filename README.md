# Lightweight Command-Line Chatbot with Model Switching

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

```python
[Qwen2.5-VL 3B] Enter your prompt:
solve this "C:\Users\alberttran\Pictures\Screenshots\Screenshot 2025-04-07 121656.png"
END 
 -> Preparing input using apply_chat_template for Qwen2.5-VL 3B...
   Found image: C:\Users\alberttran\Pictures\Screenshots\Screenshot 2025-04-07 121656.png
   Moving inputs to target device: cuda
⏳ Generating response...

💬 Response:
To solve the problem, we need to find the initial position, velocity, and acceleration of the object as functions of time \( t \), and then determine the total distance traveled by the object over the interval \( t \) on \( [0, 8] \).

### Step 1: Find the Initial Position
The initial position at time \( t = 0 \) is given by substituting \( t = 0 \) into the position function \( s(t) \):
\[
s(0) = 3(0)^3 - 40.5(0)^2 + 162(0) = 0
\]
So, the initial position is \( s(0) = 0 \).

### Step 2: Find the Velocity
The velocity \( v(t) \) is the first derivative of the position function \( s(t) \):
\[
v(t) = \frac{d}{dt} (3t^3 - 40.5t^2 + 162t) = 9t^2 - 81t + 162
\]

### Step 3: Find the Acceleration
The acceleration \( a(t) \) is the second derivative of the position function \( s(t) \):
\[
a(t) = \frac{d}{dt} (9t^2 - 81t + 162) = 18t - 81
\]

### Step 4: Discuss the Motion
We need to analyze the behavior of the velocity and acceleration functions over the interval \( t \) on \( [0, 8] \).

- **Velocity**: The velocity function \( v(t) = 9t^2 - 81t + 162 \) is a quadratic function that opens upwards. To find the critical points, we set the derivative of the velocity equal to zero:        
\[
v'(t) = 18t - 81 = 0 \implies t = \frac{81}{18} = 4.5
\]
- **Acceleration**: The acceleration function \( a(t) = 18t - 81 \) is a linear function that decreases with increasing \( t \). Its maximum value occurs at \( t = 4.5 \):
\[
a(4.5) = 18(4.5) - 81 = 81 - 81 = 0
\]

### Step 5: Determine the Total Distance Traveled
To find the total distance traveled, we need to integrate the absolute value of the velocity function over the interval \( [0, 8] \):
\[
\text{Total distance} = \int_0^8 |v(t)| \, dt
\]
Since the velocity function changes sign between \( t = 0 \) and \( t = 4.5 \), we need to split the integral at \( t = 4.5 \):
\[
|v(t)| =
\begin{cases}
9t^2 - 81t + 162 & \text{for } t \in [0, 4.5] \\
-(9t^2 - 81t + 162) & \text{for } t \in [4.5, 8]
\end{cases}
\]
Thus,
\[
\text{Total distance} = \int_0^{4.5} (9t^2 - 81t + 162) \, dt + \int_{4.5}^8 -(9t^2 - 81t + 162) \, dt
\]
Simplify the integrals:
\[
\int_0^{4.5} (9t^2 - 81t + 162) \, dt = \left[ 3t^3 - \frac{81}{2}t^2 + 162t \right]_0^{4.5} = 3(4.5)^3 - \frac{81}{2}(4.5)^2 + 162(4.5)
\]
\[
= 3 \cdot 91.125 - \frac{81}{2} \cdot 20.25 + 729 = 273.375 - 81.375 + 729 = 901
\]
\[
\int_{4.5}^8 -(9t^2 - 81t + 162) \, dt = -\left[ 3t^3 - \frac{81}{2}t^2 + 162t \right]_{4.5}^8 = -\left( 3(8)^3 - \frac{81}{2}(8)^2 + 162(8) - (3(4.5)^3 - \frac{81}{2}(4.5)^2 + 162(4.5) ) \right)      
\]
\[
= -\left( 3 \cdot 512 - \frac{81}{2} \cdot 64 + 1296 - (273.375 - 81.375 + 729) \right) = -\left( 1536 - 259.2 + 1296 - 901.375 \right)
\]
\[
= -\left( 2530.75 - 901.375 \right) = -1629.375
\]
Combine the results:
\[
\text{Total distance} = 901 - 1629.375 = 1629.375
\]

Therefore, the total distance traveled by the object over the interval \( t \) on \( [0, 8] \) is \(\boxed{1629.375}\).

🕒 Time: 106.28s | Tokens: 1321 | ⚡ Tokens/sec: 12.43
   Input Length: 478 | Output Length: 1321
   Model Type: image_text | Model Name: Qwen2.5-VL 3B
   Device: cuda | Model Device: cuda:0

[Qwen2.5-VL 3B] Enter your prompt:
exit
👋 Exiting chatbot.

Cleaning up GPU memory...
Cleanup complete.
(.venv) PS F:\qwen2.5vl> python qwen2.5.py
Using device: cuda
Select a model:
1 - Qwen2.5-VL 3B
2 - Qwen2.5-VL 7B
3 - Gemma 3 4B
4 - SmolDocling 256M
Your choice: 2

🔄 Loading model: Qwen2.5-VL 7B ...
 -> Using original loading logic for Qwen2.5-VL 7B
✅ Qwen2.5-VL 7B loaded.

   Model parameter device detected: cuda:0
=== Chatbot (Qwen2.5-VL 7B) ===
Type 'exit' to quit, 'switch' to change models. Type 'END' to submit multi-line input.


[Qwen2.5-VL 7B] Enter your prompt:
can you explain what's going on "C:\Users\alberttran\Pictures\Screenshots\Screenshot 2025-04-06 124706.png"
END 
 -> Preparing input using apply_chat_template for Qwen2.5-VL 7B...
   Found image: C:\Users\alberttran\Pictures\Screenshots\Screenshot 2025-04-06 124706.png
   Moving inputs to target device: cuda
⏳ Generating response...

💬 Response:
The image appears to be a slide from a presentation about using reinforcement learning (RL) for training models, specifically focusing on the use of verifiable rewards and classical RL techniques like PPO (Proximal Policy Optimization). Here’s a breakdown of the key points:

### Key Points:
1. **Gold final answers or verifiable constraints:**
   - The model is trained with gold final answers or verifiable constraints as the primary focus. This means that the model is evaluated based on whether it produces the correct final answer.
   - Intermediate chains of thoughts or results that do not match these constraints are not considered.

2. **Classical RL with PPO optimization:**
   - The method uses traditional RL approaches but optimizes the policy using PPO, which is a specific algorithm for improving the performance of a policy in terms of maximizing rewards.

3. **Three datasets used:**
   - The method was tested using three different datasets, which likely represent different types of problems or tasks for the model to learn from.

### Diagram Explanation:
- **Training Data:** The process starts with training data.
- **Prompts:** These prompts are the inputs to the model.
- **Policy πθ(·):** This represents the learned policy function that maps states to actions. The policy update formula suggests an iterative process where the parameters θ are updated based on the gradient J(πθ) of the policy.
- **Verifiable Reward:** The reward system rewards the model if the output matches the gold final answer (α) or 0 otherwise.
- **Scalar Reward ri:** This is a scalar value representing the reward for a specific action ai taken in the state si.
- **Completions:** These are the outputs generated by the model based on the prompts and the current policy.

### Table Explanation:
- **Prompt Dataset, Count, Verification:**
  - **GSM8K Train:** 7,473 examples, verified by exact match against extracted answers.
  - **MATH Train:** 7,500 examples, also verified by exact match against extracted answers.
  - **IF Verifiable:** 14,973 examples, verified using prompt-specific verifiers.
  - **Total:** Summarizing all examples across the datasets.

### Overall:
The slide outlines a method for training models using RL with PPO optimization, where the model is rewarded based on whether its outputs match known correct answers or satisfy certain constraints. The use of verifiable rewards ensures that the model learns to produce correct outputs. The diagram and table provide a visual representation of how this process works and the specific datasets used for training.

🕒 Time: 235.31s | Tokens: 535 | ⚡ Tokens/sec: 2.27
   Input Length: 1484 | Output Length: 535
   Model Type: image_text | Model Name: Qwen2.5-VL 7B
   Device: cuda | Model Device: cuda:0

[Qwen2.5-VL 7B] Enter your prompt:
exit 
👋 Exiting chatbot.

Cleaning up GPU memory...
Cleanup complete.
(.venv) PS F:\qwen2.5vl>


----Instruction-following demo ----
[Qwen2.5-VL 7B] Enter your prompt:
complete this program that satisfy the requirements and templates in this question "F:\qwen2.5vl\resized_image.png"
END 
 -> Preparing input using apply_chat_template for Qwen2.5-VL 7B...
   Found image: F:\qwen2.5vl\resized_image.png
   Moving inputs to target device: cuda
⏳ Generating response...

💬 Response:
To complete the program, we need to write a function that takes in a string of characters from the user using the `getstr` function provided as a hint and then prints out the entered string.

Here's the complete program:

```c
#include <stdio.h>

// Function to read a string from the user
void getstr(char s[], int max) {
    int i, c;
    i = 0;
    while ((c = getchar()) != '\n' && c != EOF)
        if (i < max)
            s[i++] = (char) c;
    s[i] = '\0';
}

int main() {
    char str[100]; // Assuming the maximum length of the string is 99 characters + null terminator

    // Call the getstr function to read the string
    printf("Enter a string: ");
    getstr(str, sizeof(str));

    // Print the entered string
    printf("You entered: %s\n", str);

    return 0;
}

### Explanation:
1. **Function `getstr`:**
   - This function reads characters from the standard input until it encounters a newline character (`'\n'`) or the end-of-file indicator (`EOF`).
   - It stops reading when either condition is met.
   - The function appends a null byte (`'\0'`) at the end of the array to mark the end of the string.

2. **Main Function:**
   - A character array `str` is declared with a maximum size of 100 bytes to accommodate the entered string plus the null terminator.
   - The `printf` function prompts the user to enter a string.
   - The `getstr` function is called to read the string from the user.
   - Finally, the entered string is printed using another `printf` statement.

This program ensures that the user can enter a string, which is then stored in the `str` array, and the contents of the array are printed to the console.

🕒 Time: 185.73s | Tokens: 416 | ⚡ Tokens/sec: 2.24
   Input Length: 357 | Output Length: 416
   Model Type: image_text | Model Name: Qwen2.5-VL 7B
   Device: cuda | Model Device: cuda:0

[Qwen2.5-VL 7B] Enter your prompt:
explain k-means algorithm to a high-schooler
END 
 -> Preparing input using apply_chat_template for Qwen2.5-VL 7B...
   Moving inputs to target device: cuda
⏳ Generating response...

💬 Response:
Sure! Let's imagine you have a bunch of colored marbles, and you want to group them into clusters based on their color. The K-means algorithm is like a fun way to do that!       

Here’s how it works:

1. **Choose the number of clusters (K):** First, decide how many different groups or "clusters" you want to make. For example, you might decide to put all the marbles into 3 groups.

2. **Randomly pick some marbles as your starting points:** These marbles will be the "centers" or "means" of each cluster. In the real world, these centers could be represented by numbers, but for marbles, they’re just the actual marbles themselves!

3. **Assign marbles to the closest center:** Now, look at all the marbles and see which one is the closest to each of your centers. Then, put all the marbles that are closest to the first center in a group together, those closest to the second center into another group, and so on. This means you’ve grouped your marbles into clusters based on their closeness.

4. **Move the centers:** Now, take a look at all the marbles in each cluster. The new center for each cluster should be right in the middle of all the marbles in that cluster. So, move each center to where the average marble of its cluster would be if there were no other marbles nearby.

5. **Repeat steps 3 and 4 until things don’t change much anymore:** Keep doing this until the centers stop moving much when you calculate the new ones. When this happens, you’ve found your clusters!

6. **Look at your clusters:** Now, you can see all your marbles sorted into neat groups based on their colors!

That’s the K-means algorithm in a nutshell! It’s a simple way to organize things into groups based on how similar they are to each other.

🕒 Time: 141.62s | Tokens: 399 | ⚡ Tokens/sec: 2.82
   Input Length: 29 | Output Length: 399
   Model Type: image_text | Model Name: Qwen2.5-VL 7B
   Device: cuda | Model Device: cuda:0
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
