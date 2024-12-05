import subprocess

# Paths to individual scripts
scripts = [
    "scripts/download_fietje.py",
    "scripts/download_geitje.py",
    "scripts/download_mistral.py"
]

def run_script(script_path):
    """
    Run a Python script using subprocess and handle errors gracefully.
    """
    try:
        print(f"Running {script_path}...")
        subprocess.run(["python", script_path], check=True)
        print(f"Completed {script_path}\n")
    except subprocess.CalledProcessError as e:
        print(f"Error while running {script_path}: {e}\n")

# Execute all scripts
for script in scripts:
    run_script(script)

print("All downloads completed.")
