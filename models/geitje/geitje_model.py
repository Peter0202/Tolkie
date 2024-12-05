from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from accelerate import infer_auto_device_map
import torch

app = Flask(__name__)

# Path to the locally saved model
model_name = "./model"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model with device map...")

# Step 1: Temporarily initialize the model to infer device_map
model_dummy = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True  # Reduce CPU memory usage during loading
)

device_map = infer_auto_device_map(
    model_dummy,
    no_split_module_classes=["LlamaDecoderLayer"],  # Use your model's architecture
    max_memory={0: "12GiB", "cpu": "48GiB"}  # Adjust memory allocation
)

# Step 2: Load the actual model with the device_map
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map=device_map,
    low_cpu_mem_usage=True,
    offload_folder="./offload"  # Folder for offloading layers if memory is exceeded
)

# Log device placement for debugging
for name, param in model.named_parameters():
    print(f"{name}: {param.device}")

# Initialize the text generation pipeline
print("Initializing text-generation pipeline...")
text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    truncation=True  # Enable truncation to prevent token overflow
)

def sliding_window_chunks(text, tokenizer, max_tokens=2048, overlap=512):
    """
    Splits the text into overlapping chunks of `max_tokens` size.
    """
    tokens = tokenizer.encode(text, truncation=False)
    print(f"Total tokens in input: {len(tokens)}")  # Debugging log

    if len(tokens) <= max_tokens:
        print("Input fits within max_tokens. No need for sliding window.")
        return [tokens]  # Return as a single chunk

    chunks = []
    for i in range(0, len(tokens), max_tokens - overlap):
        chunk = tokens[i:i + max_tokens]
        chunks.append(chunk)

    for idx, chunk in enumerate(chunks):
        print(f"Chunk {idx + 1}/{len(chunks)}: {len(chunk)} tokens")
    return chunks

@app.route('/generate', methods=['POST'])
def generate():
    """
    Handles text generation using sliding windows to fit within the model's token limit.
    """
    data = request.json
    prompt = data.get("prompt", "")

    try:
        max_tokens = 2048  # Model's maximum token capacity
        overlap = 512  # Overlap for context preservation
        chunks = sliding_window_chunks(prompt, tokenizer, max_tokens, overlap)

        responses = []
        for chunk in chunks:
            chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
            print(f"Processing chunk (tokens: {len(chunk)}): {chunk_text}...")

            # Generate text for the chunk
            generated_text = text_generator(
                chunk_text,
                max_new_tokens=512,
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                temperature=0.7,
                no_repeat_ngram_size=3
            )[0]["generated_text"]

            responses.append(generated_text)

        combined_response = " ".join(responses)
        print(f"Combined response length (chars): {len(combined_response)}")
        return jsonify({"response": combined_response})

    except Exception as e:
        error_response = {"error": str(e)}
        print(f"Error response: {error_response}")
        return jsonify(error_response), 500


if __name__ == '__main__':
    # Run the Flask server
    app.run(host='0.0.0.0', port=8002)
