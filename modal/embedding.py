import modal
from server import modal_app, ColPaliServer, GPU_CONCURRENCY
import json
import base64
from io import BytesIO
import logging
import time
import os
from dataset import get_collection_dataset_names

logger = logging.getLogger(__name__)

BATCH_SIZE = 5
DATASET_DIR = "/data"
CHECKPOINT_DIR = "/checkpoint"
DATASET_READ_VOLUME  = modal.Volume.from_name("colpali-dataset")
EMBEDDING_CHECKPOINT_VOLUME = modal.Volume.from_name(
    "colpali-embedding-checkpoint", create_if_missing=True
)

@modal_app.function(
    image=modal.Image.debian_slim().pip_install(
        "datasets", "huggingface_hub", "fastapi", "httpx", "Pillow", "pyarrow", "msgspec"
    ),
    volumes={
        DATASET_DIR: DATASET_READ_VOLUME,
        CHECKPOINT_DIR: EMBEDDING_CHECKPOINT_VOLUME
    },
    cpu=2,
    timeout=86400,
)
def embed_dataset(down_scale: float = 1.0, batch_size: int = BATCH_SIZE):
    """Embed dataset using Modal volume"""
    import os
    import json
    from datasets import load_from_disk
    import time
    from PIL import Image

    def convert_pil_to_b64_image(image: Image.Image) -> str:
        """
        Convert a PIL Image to a base64 string.
        """
        image_bytes = BytesIO()
        image.save(image_bytes, format="PNG")
        image_base64 = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{image_base64}"

    # Create model client
    colpali = ColPaliServer()
    logger.info("ColPali server started successfully")

    # Load dataset
    collection_name = "vidore/vidore-benchmark-667173f98e70a1c0fa4db00d"
    dataset_names = get_collection_dataset_names(collection_name) 
    # dataset_names.extend(["ufo-ColPali", "DocVQA"])

    start = time.perf_counter()
    for dataset_name in dataset_names:
        final_path = f"{CHECKPOINT_DIR}/{dataset_name}"
        if os.path.exists(final_path):
            logger.info(f"Dataset {dataset_name} already processed")
            continue
            
        dataset_path = os.path.join(DATASET_DIR, dataset_name)
        dataset = load_from_disk(dataset_path)

        logger.info(f"Dataset {dataset_name} loaded successfully: {len(dataset)} samples")
        # Process data
        if "/" in dataset_name:
            dataset_name = dataset_name.split("/")[-1]  
        checkpoint_file = os.path.join(CHECKPOINT_DIR, f"{dataset_name}_embedding_checkpoint_info.json")
        
        # Check if checkpoint exists
        completed_ids = set()
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, "r") as f:
                checkpoint_data = json.load(f)
                completed_ids = set(checkpoint_data.get("completed_ids", []))
        
        # Prepare batch processing
        batches = []
        batch = 0
        for idx, item in enumerate(dataset):
            if idx % int(1/down_scale) != 0:
                if idx % batch_size == 0:
                    batch += 1
                continue
            if idx in completed_ids:
                if idx % batch_size == 0:
                    batch += 1
                continue
            
            if item.get('query') == None:
                continue
            chunk = {
                "id": idx, 
                "image": convert_pil_to_b64_image(item['image']) ,
                "query": item['query'] # davanstrien/ufo-ColPali : raw_query; lmms-lab/DocVQA: question
            }
            batches.append(chunk)
            
            if len(batches) >= batch_size:
                process_batch_data(batches, colpali, completed_ids, batch,checkpoint_file)
                batch += 1
                batches = []
        
        # Process remaining batches
        if batches:
            process_batch_data(batches, colpali, completed_ids, batch,checkpoint_file)
        
        save_dataset_to_final_checkpoint(dataset_name, final_path)
        os.remove(checkpoint_file)
        logger.info(f"Dataset {dataset_name} processing completed and merged to {final_path}")
    
    end = time.perf_counter()
    duration = end - start
    resp = {
        "downscale": down_scale,
        "batch_size": batch_size,
        "n_gpu": GPU_CONCURRENCY,
        "duration_mins": duration / 60,
    }

    return resp

def save_dataset_to_intermediate_checkpoint(embeddings, filename, batch):
    """Saves the dataset to an intermediate checkpoint.
    """
    """Save checkpoint in parquet format"""
    import pandas as pd
    
    # Prepare data
    data = {
        'id': [],
        'image_embedding': [],
        'image_embedding_shape': [],
        'query': [],
        'query_embedding': [],
        'query_embedding_shape': [],
    }
    
    for r in embeddings:
        image_emb = r["image_embedding"]
        query_emb = r["query_embedding"]
        data['id'].append(r["id"])
        data['image_embedding'].append(image_emb.tobytes())  # Serialize numpy array
        data['image_embedding_shape'].append(image_emb.shape)
        data['query'].append(r["query"])
        data['query_embedding'].append(query_emb.tobytes())
        data['query_embedding_shape'].append(query_emb.shape)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Build output path
    filename = filename.split("_embedding_checkpoint_info.json")[0]
    checkpoint_path = f"{filename}_batch_{batch}.parquet"
    
    # Save parquet file
    df.to_parquet(checkpoint_path, index=False)
    logger.info(f"Saved checkpoint to {checkpoint_path}")

def save_dataset_to_final_checkpoint(dataset_name,final_path):
    """Merge parquet files into Dataset"""
    import pandas as pd
    import pyarrow as pa
    from datasets import Dataset
    import glob
    import numpy as np
    
    # Find all parquet files
    parquet_files = sorted(glob.glob(f"{CHECKPOINT_DIR}/{dataset_name}_batch_*.parquet"),
                          key=lambda x: int(x.split('_batch_')[-1].split('.')[0]))
    
    # Read and merge all DataFrames
    dfs = []
    for pq_file in parquet_files:
        try:
            df = pd.read_parquet(pq_file)
            dfs.append(df)
            logger.info(f"Loaded {pq_file}")
        except Exception as e:
            logger.error(f"Failed to load {pq_file}: {e}")
            continue
    
    if not dfs:
        return None
        
    # Merge DataFrames
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Create pyarrow table
    table = pa.Table.from_arrays(
        [
            pa.array(merged_df['id'].tolist()),
            pa.array(merged_df['image_embedding']),
            pa.array(merged_df['image_embedding_shape'].tolist()),
            pa.array(merged_df['query_embedding']),
            pa.array(merged_df['query_embedding_shape'].tolist()),
            pa.array(merged_df['query'].tolist()),
        ],
        names=['id', 'image_embedding', 'image_embedding_shape', 'query_embedding', 'query_embedding_shape', 'query']
    )
    
    # Save as Dataset
    dataset = Dataset(table)
    dataset.save_to_disk(final_path)
    
    # Clean up parquet files
    for pq_file in parquet_files:
        try:
            os.remove(pq_file)
            logger.info(f"Removed {pq_file}")
        except Exception as e:
            logger.warning(f"Failed to remove {pq_file}: {e}")
    return 

def process_batch_data(batches, colpali: ColPaliServer, completed_ids, batch, checkpoint_file):
    """Process batch and update checkpoint"""
    image_embeddings = colpali.embed_images.remote([batch["image"] for batch in batches])
    query_embeddings = colpali.embed_queries.remote([batch["query"] for batch in batches])
    # Save results
    batch_results = []
    for i, b in enumerate(batches):
        image_embedding = image_embeddings[i]
        query_embedding = query_embeddings[i]
        id = b["id"]
        batch_results.append({
            "id": id,
            "image_embedding": image_embedding,
            "query": b["query"],
            "query_embedding": query_embedding, 
        })
        completed_ids.add(id)

    save_dataset_to_intermediate_checkpoint(batch_results, checkpoint_file, batch)
     # Update checkpoint
    with open(checkpoint_file, "w") as f:
        json.dump({
            "completed_ids": list(completed_ids),
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f)
    EMBEDDING_CHECKPOINT_VOLUME.commit()

