from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from accelerate import infer_auto_device_map
import torch

# Path to the locally saved model
model_name = "./model"

# Function to print the prompt and generate a response
def run_inference(prompt):
    print(f"Prompt:\n{prompt}\n")

    # Split the input text into manageable chunks
    max_tokens = 2048  # Model's maximum token capacity
    overlap = 512  # Overlap for context preservation
    chunks = sliding_window_chunks(prompt, tokenizer, max_tokens, overlap)

    responses = []
    for chunk in chunks:
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
        print(f"Processing chunk (tokens: {len(chunk)}): {chunk_text}\n")

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
    print(f"Response:\n{combined_response}\n")
    return combined_response


# Sliding window function
def sliding_window_chunks(text, tokenizer, max_tokens=2048, overlap=512):
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

# Input your test prompt
test_prompt = """
### Instructie
Je bent specialist op het gebied van teksten samenvatten voor laaggeletterden. Al jouw samenvattingen zijn geschreven op taalniveau A2. Je samenvattingen zijn volledig, zodat de uiteindelijke lezer de strekking van de tekst terugvindt in jouw samenvatting. Baseer je volledig op de tekst die ik aan je geef, zonder dat je hier zelf andere informatie bijhaalt.

1. Geef van de onderstaande tekst 1 korte zin die beschrijft waar de tekst over gaat:
2. Geef van onderstaande tekst 2 korte zinnen die de belangrijkste punten uit de tekst benoemen:
3. Herschrijf de zinnen uit stap 1 en 2 volgens de ‘Tekstvoorwaarden voor laaggeletterden’ die ik je hieronder geef.
4. Kijk nog eens goed of de zinnen uit stap 3 voldoen aan de ‘Tekstvoorwaarden voor laaggeletterden. Zo niet, herschrijf ze dan. De zinnen moeten kort zijn (het liefst maximaal 10 woorden) en mogen geen moeilijke woorden bevatten.
5. Geef als antwoord alleen de herschreven zinnen uit stap 4

### Tekstvoorwaarden voor laaggeletterden
- Schrijf getallen altijd als cijfer en niet als woord
- Vermijd spreekwoorden, uitdrukkingen en gezegden
- Vermijd tangconstructies
- Vermijd passieve zinnen
- Schrijf alleen korte zinnen. Iedere gedachte moet in een aparte zin staan. Het liefst schrijf je zinnen van maximaal 5 woorden. Breek een zin die langer is dan 5 woorden snel op in kleinere zinnen.
- Gebruik alleen hele eenvoudige woorden.
- Als je toch een heel moeilijk woord moet gebruiken, leg je het uit
- Gebruik geen hij of zij als je niet weet wat het geslacht van een persoon is. Gebruik in dat geval de functie of rol van de persoon
- Gebruik zoveel mogelijk dezelfde woorden
- Je mag zinnen beginnen met ‘En’ of ‘Maar’

### **Input**
De input die je krijgt is:

Over longgeneeskunde Of het nu gaat om astma, COPD, bronchiëctasieën of een uitgebreid revalidatieprogramma: op de polikliniek longgeneeskunde van Máxima MC (MMC) bieden we de beste zorg voor u. De poli bevindt zich zowel in Eindhoven als Veldhoven. Onze longartsen zorgen voor snelle toegankelijke en persoonlijke begeleiding. Alle specialisten kennen hun eigen aandachtsgebied, waardoor zij u met specifieke aandoeningen goed kunnen helpen. Er zijn korte lijnen met huisartsen, fysiotherapeuten, diëtisten, longverpleegkundigen en specialisten, waardoor we een multidisciplinaire behandeling aanbieden. Daarnaast werken we ook veel samen met universitaire centra. Op deze website vindt u meer informatie over aandoeningen, symptomen, oorzaken, behandelingen en hoe u een afspraak maakt op de polikliniek longgeneeskunde."""

# Run the inference
run_inference(test_prompt)
