import os
import requests
from flask import Flask, request, jsonify
from transformers import AutoTokenizer
import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Set logging level to DEBUG for detailed output
    format="%(asctime)s [%(levelname)s] %(message)s",  # Include timestamps in logs
    handlers=[
        logging.StreamHandler()  # Output logs to the console
    ]
)
logger = logging.getLogger(__name__)

# Define the paths for each model
MODEL_PATHS = {
    "fietje": "./models/fietje/model",
    "geitje": "./models/geitje/model"
}

app = Flask(__name__)

def load_prompt(model_name, prompt_version):
    """
    Load the appropriate prompt for the given model and version.
    """
    prompt_folder = "prompts/"
    prompt_file = f"prompt_v{prompt_version}_{model_name}.txt"
    prompt_path = os.path.join(prompt_folder, prompt_file)
    logger.debug(f"Looking for model-specific prompt file at: {prompt_path}")

    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as file:
            return file.read().strip()

    # Fallback to universal prompt
    universal_prompt_file = f"prompt_v{prompt_version}_universal.txt"
    universal_prompt_path = os.path.join(prompt_folder, universal_prompt_file)
    logger.warning(f"Model-specific prompt not found. Falling back to universal prompt at: {universal_prompt_path}")

    if os.path.exists(universal_prompt_path):
        with open(universal_prompt_path, "r", encoding="utf-8") as file:
            return file.read().strip()

    # No prompt found
    logger.error(f"No prompt found for model '{model_name}' and version '{prompt_version}'")
    return None

def chunk_text(text, tokenizer, max_tokens=512, overlap=100):
    """
    Split input text into manageable chunks based on token limit.
    """
    logger.debug("Chunking text for processing...")
    encoded = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(encoded), max_tokens - overlap):
        chunk = encoded[i:i + max_tokens]
        chunks.append(tokenizer.decode(chunk, skip_special_tokens=True))
    logger.debug(f"Generated {len(chunks)} chunks.")
    return chunks

def send_prompt_to_model(prompt_text, model_choice):
    """
    Sends the prompt to the model's Flask endpoint and returns the response.
    """
    model_url = f"http://localhost:8003/generate" if model_choice == "fietje" else f"http://localhost:8002/generate"
    payload = {"prompt": prompt_text}
    logger.info(f"Sending request to {model_url} with payload length: {len(prompt_text)} characters")
    try:
        response = requests.post(model_url, json=payload)
        response.raise_for_status()
        response_json = response.json()
        logger.debug(f"Model response: {response_json}")
        return response_json.get("response")
    except Exception as e:
        logger.error(f"Error while sending request to model: {e}", exc_info=True)
        return None

@app.route('/generate', methods=['POST'])
def generate():
    """
    Main API endpoint to process text and interact with the model.
    """
    logger.info("Received request at /generate endpoint.")
    try:
        # Extract parameters from request
        data = request.json
        logger.debug(f"Request JSON: {data}")

        # Validate model choice dynamically
        model_choice = data.get("model", "").strip().lower()
        if not model_choice or model_choice not in MODEL_PATHS:
            error_msg = f"Invalid or missing model choice: {model_choice}. Must be one of {list(MODEL_PATHS.keys())}."
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 400

        prompt_version = data.get("prompt_version", "0")  # Default version is 0
        custom_text = data.get("text", "")

        # Load the correct tokenizer dynamically based on model choice
        model_path = MODEL_PATHS[model_choice]
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load the correct prompt
        default_prompt = load_prompt(model_choice, prompt_version)
        if not default_prompt:
            error_msg = f"No prompt found for model '{model_choice}' and version '{prompt_version}'."
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 404

        # Combine prompt with text
        combined_prompt = f"{default_prompt}\n\n{custom_text}"
        logger.debug(f"Combined prompt: {combined_prompt[:200]}...")

        # Send prompt to the correct model endpoint
        translated_text = send_prompt_to_model(combined_prompt, model_choice)
        if translated_text:
            logger.info("Response successfully generated.")
            return jsonify({"response": translated_text})
        else:
            logger.error("Failed to get a response from the model.")
            return jsonify({"error": "Failed to get a response from the model."}), 500

    except Exception as e:
        logger.error(f"Unexpected error in /generate: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
