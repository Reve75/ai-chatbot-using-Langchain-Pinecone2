from transformers import AutoModel, AutoTokenizer
import os

# Specify the model name and local directory to save it
model_name = "sentence-transformers/all-MiniLM-L6-v2"
local_dir = "./all-MiniLM-L6-v2"

# Make sure the directory exists
os.makedirs(local_dir, exist_ok=True)

# Download and save the model and tokenizer
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save the model and tokenizer to the specified directory
model.save_pretrained(local_dir)
tokenizer.save_pretrained(local_dir)

print(f"Model and tokenizer downloaded and saved to {local_dir}")
