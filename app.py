import os
import requests
from flask import Flask, request, jsonify
from transformers import AutoTokenizer
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# define paths
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

    logger.error(f"No prompt found for model '{model_name}' and version '{prompt_version}'")
    return None

def send_prompt_to_model(prompt_text, assistant_part, model_choice):
    """
    Sends the prompt to the model's Flask endpoint and returns the response.
    """
    # Decide which model endpoint to call
    model_url = "http://localhost:8003/generate" if model_choice == "fietje" else "http://localhost:8002/generate"

    # Combine final prompt if assistant_part is present
    final_prompt = f"{prompt_text}\n\n{assistant_part}" if assistant_part else prompt_text

    payload = {"prompt": final_prompt}
    logger.info(f"Sending request to {model_url} with payload length: {len(final_prompt)} characters")

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
        # extract parameters
        data = request.json
        logger.debug(f"Request JSON: {data}")

        # validate model choice
        model_choice = data.get("model", "").strip().lower()
        if not model_choice or model_choice not in MODEL_PATHS:
            error_msg = f"Invalid or missing model choice: {model_choice}. Must be one of {list(MODEL_PATHS.keys())}."
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 400

        prompt_version = data.get("prompt_version", "0")  # Default version is 0
        custom_text = data.get("text", "")
        assistant_part = data.get("assistant", "")  # <-- new field from your JSON

        # load the correct tokenizer based on model choice
        model_path = MODEL_PATHS[model_choice]
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # load the correct prompt
        default_prompt = load_prompt(model_choice, prompt_version)
        if not default_prompt:
            error_msg = f"No prompt found for model '{model_choice}' and version '{prompt_version}'."
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 404

        # combine prompt with text (but do not yet attach assistant_part)
        combined_prompt = f"{default_prompt}\n\n{custom_text}"
        logger.debug(f"Combined prompt: {combined_prompt[:200]}...")

        # send prompt + assistant part to correct model endpoint
        translated_text = send_prompt_to_model(combined_prompt, assistant_part, model_choice)
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
