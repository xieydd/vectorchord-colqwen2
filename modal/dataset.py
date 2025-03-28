import modal
from typing import List
import logging

logger = logging.getLogger(__name__)

image = modal.Image.debian_slim().pip_install("datasets","huggingface_hub","Pillow")
DATASET_DIR = "/data"
DATASET_VOLUME  = modal.Volume.from_name(
    "colpali-dataset", create_if_missing=True
)
app = modal.App(image=image)
@app.function(volumes={DATASET_DIR: DATASET_VOLUME}, timeout=3000)
def download_dataset(cache=False) -> None:
    from datasets import load_dataset
    from tqdm import tqdm

    
    collection_dataset_names = get_collection_dataset_names("vidore/vidore-benchmark-667173f98e70a1c0fa4db00d") 
    for dataset_name in tqdm(collection_dataset_names, desc="vidore benchmark dataset(s)"):
        dataset = load_dataset(dataset_name, split="test",num_proc=10)
        unique_indices = dataset.to_pandas().drop_duplicates(subset="image_filename", keep="first").index #to remove repeating PDF pages with different queries
        dataset = dataset.select(unique_indices)
        dataset.save_to_disk(f"{DATASET_DIR}/{dataset_name}")

    # dataset = load_dataset("davanstrien/ufo-ColPali", split="train",num_proc=10)
    # dataset = dataset.filter(lambda x: x['parsed_into_json'])
    # dataset.save_to_disk(f"{cache_dir}/ufo-ColPali")

    # dataset = load_dataset("lmms-lab/DocVQA", "DocVQA",split="validation", num_proc=10)
    # dataset.save_to_disk(f"{cache_dir}/DocVQA")
    # Commit and save to the volume
    DATASET_VOLUME.commit()

def get_collection_dataset_names(collection_name: str) -> List[str]:
    import huggingface_hub
    collection_name = "vidore/vidore-benchmark-667173f98e70a1c0fa4db00d"
    logger.info(f'Loading datasets from the Hf Hub collection: "{collection_name}"')
    collection = huggingface_hub.get_collection(collection_name)
    collection_dataset_names = [dataset_item.item_id for dataset_item in collection.items]
    return collection_dataset_names