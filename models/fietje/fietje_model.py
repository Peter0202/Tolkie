from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

# Load the model and tokenizer
model_name = "./model"  # Path to the locally saved model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
# model.resize_position_embeddings(4096)  # Match the new `max_position_embeddings`
# model.save_pretrained("./model")  # Save the updated model

# Initialize the text generation pipeline
text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0,  # Use GPU if available; use -1 for CPU
    truncation=True  # Enable truncation to prevent token overflow
)

def sliding_window_chunks(text, tokenizer, max_tokens=2048, overlap=512):
    """
    Splits the text into overlapping chunks of `max_tokens` size.
    """
    # Tokenize the input text
    tokens = tokenizer.encode(text, truncation=False)

    # Validate token size before processing
    print(f"Total tokens in input: {len(tokens)}")  # Debugging log
    if len(tokens) <= max_tokens:
        print("Input fits within max_tokens. No need for sliding window.")
        return [tokens]  # Return as a single chunk

    # Create chunks with overlap
    chunks = []
    for i in range(0, len(tokens), max_tokens - overlap):
        chunk = tokens[i:i + max_tokens]
        chunks.append(chunk)

    # Debugging log for chunk sizes
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
        # Split the input text into manageable chunks
        max_tokens = 2048  # Model's maximum token capacity
        overlap = 512  # Overlap for context preservation
        chunks = sliding_window_chunks(prompt, tokenizer, max_tokens=2048, overlap=512)

        # Debugging: Ensure each chunk size is within the limit
        for idx, chunk in enumerate(chunks):
            print(f"Chunk {idx + 1} size: {len(chunk)} tokens")

        # Log total tokens and chunk sizes for debugging
        total_tokens = tokenizer.encode(prompt, truncation=False)
        print(f"Total input tokens: {len(total_tokens)}")
        responses = []

        # Generate responses for each chunk
        for chunk in chunks:
            # Decode the tokenized chunk into text
            chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
            print(f"Processing chunk (tokens: {len(chunk)}): {chunk_text[:100]}...")

            # Generate text for the chunk
            generated_text = text_generator(
                chunk_text,  # Pass the decoded text to the generator
                max_new_tokens=512,  # Limit output size per chunk
                num_return_sequences=1,
                do_sample=True,  # Enable sampling for variability
                top_k=50,  # Sample from the top 50 tokens
                temperature=0.7,  # Adjust randomness
                no_repeat_ngram_size=3  # Avoid repetitive phrases
            )[0]["generated_text"]

            responses.append(generated_text)

        # Combine responses into a single output
        combined_response = " ".join(responses)
        print(f"Combined response length (chars): {len(combined_response)}")
        return jsonify({"response": combined_response})

    except Exception as e:
        # Handle exceptions and return error message
        error_response = {"error": str(e)}
        print(f"Error response: {error_response}")
        return jsonify(error_response), 500


if __name__ == '__main__':
    # Run the Flask server
    app.run(host='0.0.0.0', port=8003)

