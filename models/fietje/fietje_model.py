from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

# load model
model_name = "./model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
# model.resize_position_embeddings(4096)
# model.save_pretrained("./model")

# init pipeline
text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0,
    truncation=True
)

def sliding_window_chunks(text, tokenizer, max_tokens=2048, overlap=512):
    """
    Splits the text into overlapping chunks of `max_tokens` size.
    """
    # tokenize input 
    tokens = tokenizer.encode(text, truncation=False)

    # validate token size before processing
    print(f"Total tokens in input: {len(tokens)}")
    if len(tokens) <= max_tokens:
        print("Input fits within max_tokens. No need for sliding window.")
        return [tokens]  # return single chunk

    # create chunks w/ overlap
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
        # split input into chunks
        max_tokens = 2048 
        overlap = 512
        chunks = sliding_window_chunks(prompt, tokenizer, max_tokens=2048, overlap=512)

        for idx, chunk in enumerate(chunks):
            print(f"Chunk {idx + 1} size: {len(chunk)} tokens")

        total_tokens = tokenizer.encode(prompt, truncation=False)
        print(f"Total input tokens: {len(total_tokens)}")
        responses = []

        # generate response for each chunk
        for chunk in chunks:
            chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
            print(f"Processing chunk (tokens: {len(chunk)}): {chunk_text[:100]}...")

            # generate text for the chunk
            generated_text = text_generator(
                chunk_text,
                max_new_tokens=300,
                min_length=50,
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                top_p=0.7,
                temperature=0.2,
                no_repeat_ngram_size=4,
                repetition_penalty=1.2
            )[0]["generated_text"]



            responses.append(generated_text)

        # combine responses
        combined_response = " ".join(responses)
        print(f"Combined response length (chars): {len(combined_response)}")
        return jsonify({"response": combined_response})

    except Exception as e:
        error_response = {"error": str(e)}
        print(f"Error response: {error_response}")
        return jsonify(error_response), 500


if __name__ == '__main__':
    # flask server
    app.run(host='0.0.0.0', port=8003)

