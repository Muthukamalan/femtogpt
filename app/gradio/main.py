import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import os
from dotenv import load_dotenv
import gradio as gr

load_dotenv()

# Global variables
model = None
tokenizer = None

# 🔹 Load model once at startup
def load_model():
    global model, tokenizer

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m")

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Model loaded!")

load_model()


# 🔹 Generation function (used by Gradio UI)
def generate_text(prompt, max_tokens, temperature, top_k):
    if model is None:
        return "Model not loaded"

    try:
        prompt_ids = tokenizer.encode(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        device = next(model.parameters()).device
        prompt_ids = prompt_ids.to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                prompt_ids,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
                top_k=int(top_k),
            )

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text

    except Exception as e:
        return f"Error: {str(e)}"


# 🔹 Gradio UI
interface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(lines=4, placeholder="Enter your prompt..."),
        gr.Slider(10, 500, value=100, label="Max Tokens"),
        gr.Slider(0.1, 1.5, value=0.8, label="Temperature"),
        gr.Slider(1, 100, value=40, label="Top-K"),
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="Qwensmol Text Generator",
    description="Generate text using google/gemma-3-270m",
)

# 🔹 Launch app
if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)