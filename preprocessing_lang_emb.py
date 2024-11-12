import os

import torch
import yaml

from models.multimodal_encoder.t5_encoder import T5Embedder


GPU = 4
MODEL_PATH = "google/t5-v1_1-xxl"
CONFIG_PATH = "/home/1ms.ai/dora/node-hub/dora-rdt-1b/dora_rdt_1b/RoboticsDiffusionTransformer/configs/base.yaml"

dataset_dir = "/home/1ms.ai/dora/node-hub/dora-rdt-1b/dora_rdt_1b/RoboticsDiffusionTransformer/data/data/dataset/agilex/rdt_data"

# Note: if your GPU VRAM is less than 24GB, 
# it is recommanded to enable offloading by specifying an offload directory.
OFFLOAD_DIR = None  # Specify your offload directory here, ensuring the directory exists.

def main():
    with open(CONFIG_PATH, "r") as fp:
        config = yaml.safe_load(fp)
    
    device = torch.device(f"cuda:{GPU}")
    text_embedder = T5Embedder(
        from_pretrained=MODEL_PATH, 
        model_max_length=config["dataset"]["tokenizer_max_length"], 
        device=device,
        use_offload_folder=OFFLOAD_DIR
    )
    tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model

    for task_name in os.listdir(dataset_dir):
        task_dir = os.path.join(dataset_dir, task_name)
        embedding_name = os.path.join(task_dir, 'lang_embed.pt')
        if os.path.exists(embedding_name):
            continue
        task_name = " ".join(task_name.split("_"))
        tokens = tokenizer(
            task_name, return_tensors="pt",
            padding="longest",
            truncation=True
        )["input_ids"].to(device)
        
        tokens = tokens.view(1, -1)
        with torch.no_grad():
            pred = text_encoder(tokens).last_hidden_state.detach().cpu()[0]
    # We save the embeddings in a dictionary format
        torch.save(
             pred,
             embedding_name
        )
    
        print(f'\"{task_name}\" into shape {pred.shape} and saved to \"{embedding_name}\"')


if __name__ == "__main__":
    main()
