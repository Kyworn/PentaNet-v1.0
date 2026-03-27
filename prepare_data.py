import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

DATA_DIR = "data/wikitext-103"

def prepare():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("📥 Loading wikitext-103-v1 from HuggingFace datasets...")
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    
    print("📥 Loading GPT-2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    for split in ['train', 'validation', 'test']:
        output_file = os.path.join(DATA_DIR, f"{split}.bin")
        if os.path.exists(output_file):
            print(f"⏩ {output_file} already exists.")
            continue
            
        print(f"⚗️ Tokenizing {split} split...")
        all_ids = []
        # Process in chunks to give progress bar
        for text in tqdm(dataset[split]['text'], desc=split):
            if not text.strip(): 
                continue
            # Adding eos_token_id to separate lines as GPT2 generally does not encode \n perfectly or we want clean boundaries
            ids = tokenizer.encode(text) + [tokenizer.eos_token_id]
            all_ids.extend(ids)
            
        arr = np.array(all_ids, dtype=np.uint32)
        arr.tofile(output_file)
        print(f"✅ {split}.bin created: {len(arr)} tokens.")

if __name__ == '__main__':
    prepare()
