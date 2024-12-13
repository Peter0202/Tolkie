# Tolkie - Alternative Models

Introduction << write project intro here

## Getting Started

### Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop/)

### Setting up Prompts
In order to use the Tolkie models, you need to add prompt files in the prompts directory.

- Each prompt should be a `.txt` file placed in the [Prompts](./prompts/) folder.

- Naming Convention: 
    ```
    prompt_v#_model-name.txt
    ```
    - `#` should be a version number (between 0-1000) that distinguishes different prompt versions.
    - `model-name` should be one of the following:

        `fietje`: For prompts specifically designed for the Fietje model.

        `geitje`: For prompts specifically designed for the Geitje model.

        `universal`: Generic prompts suitable for either model.
    - Example: *prompt_v5_fietje.txt* would be the fifth version of a prompt intended for the fietje model.

- Ensure that at least one valid prompt file exists in the prompts folder to start the application.

> Note: Use the Tolkie-made prompts for the best results.

# Running locally
To run the application locally, first install the correct python version with libraries

### Using Chocolatey Install:
Open powershell as administrator:
```bash 
choco install python --version=3.11.6
```

### Using Installer:
Just download the installer and follow the instructions:
https://www.python.org/downloads/release/python-3116/
> Note: You might have to add a path to your python version manually using system environment variables.

### Install Python Libraries:
Open powershell as administrator:
```bash 
pip install flask transformers
pip install safetensors
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 #<<< CHANGE TO YOUR OWN VERSION
pip install accelerate
pip install sentencepiece
pip install blobfile
pip install tiktoken
pip install huggingface_hub
pip install requests
```
> Note: There might be more requirements, just install them the same way if required.

### Download Models
Navigate to the root of the project and download the models using the following commands:

```bash 
python scripts/download_fietje.py
python scripts/download_geitje.py
python scripts/download_mistral.py
python scripts/download_all_models.py
```

> Note: Download_mistral.py & download_all_models.py require huggingface account & permissions.

> Note: Just download fietje & geitje if you dont have them.

### Test it
Open powershell terminal in root:
```bash
python app.py
```

Open powershell terminal in root:
```bash
cd .\models\fietje\
python fietje_model.py
```

Open powershell terminal in root:
```bash
cd .\models\geitje\
python geitje_model.py
```

Open cmd/commandprompt terminal in root: 
```bash
curl -X POST -H "Content-Type: application/json" -d "@example.json" http://localhost:8002/generate
```
> Note: You should edit example.json for testing requests:
> - model: geitje/fietje
> - prompt_version: 0/1/... if you add more prompts
> - text: text it should test, experiment

> Note: Not more than one model at the same time, your pc will die.

# Running in Docker
To run the application with docker, run the following command:

```bash
docker compose up
```
> Note: This might take a couple of minutes the first time, as it will start downloading the LLM model.

> Note: For the Geitje & Fietje containers a GPU driver is needed. Has only been tested with a NVIDIA GPU.

## How to use
- Once all Docker Containers are up and running you can access the API and send requests. 

- The API expects a JSON file with 3 keys and values:

    `model`: Should be one of the following: `geitje` or `fietje`.

    `prompt_version`: Should be a version number of your prefered prompt version.

    `text`: Should be a string containing the text you want the model to simplify.

__Example Bash Request:__
```bash
curl -X POST http://localhost:5000/generate \
-H "Content-Type: application/json" \
-d '{
  "model": "fietje",
  "prompt_version": "0",
  "text": "Simplificeer deze tekst voor me."
}'
```
__Expected Response:__
```bash
{
  "response": "Simpelere versie van de tekst hier."
}
```

> Note: There is a [example json](./example.json) in root.
