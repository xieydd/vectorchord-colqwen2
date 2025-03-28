from typing import List
import base64
from io import BytesIO
import logging
import modal
import numpy as np

logger = logging.getLogger(__name__)

class ColPaliModel:
    def __init__(self, model_name: str = "vidore/colqwen2-v1.0", cache_dir: str="/model"):
        import torch
        self.model_name = model_name
        self.cache_dir = cache_dir
        # colpali model
        # select model from https://huggingface.co/spaces/vidore/vidore-leaderboard
        if self.model_name == "vidore/colqwen2-v1.0":
            from colpali_engine.models import ColQwen2, ColQwen2Processor
            from transformers.utils.import_utils import is_flash_attn_2_available

            model = ColQwen2.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="cuda:0", 
                attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
                cache_dir=self.cache_dir,
            ).eval()

            colpali_processor = ColQwen2Processor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
            )
        else:
            raise ValueError(f"Model {self.model_name} not supported")
        self.model = model
        self.processor = colpali_processor
    
    @staticmethod
    def decode_base64_to_image(b64_image: str):
        from PIL import Image
        """
        Convert a base64 image string to a PIL Image.
        """
        if b64_image.startswith("data:image"):
            b64_image = b64_image.split(",")[1]

        try:
            image_data = base64.b64decode(b64_image)
            image_bytes = BytesIO(image_data)
            image = Image.open(image_bytes)
        except Exception as e:
            raise ValueError("Failed to convert base64 string to PIL Image") from e

        return image

    @modal.method()  
    async def embed_image(self, image_base64: str) -> np.ndarray: 
        import torch
        """Generate embedding for a single image"""
        try:
            # Decode base64 image
            image = self.decode_base64_to_image(image_base64)
            
            # Process the image
            batch_image = self.processor.process_images([image]).to(self.model.device)
            
            # Generate embedding
            with torch.no_grad():
                embedding = self.model(**batch_image)
                
            # Convert to list
            embedding_np = embedding.cpu().to(torch.float32).detach().numpy()
            return embedding_np[0]
        except Exception as e:
            raise ValueError(f"Image embedding generation failed: {str(e)}")
    
    @modal.method()
    async def batch_embed_images(self, images_base64: List[str]) -> np.ndarray:
        import torch
        from PIL import Image
        """Generate embeddings for multiple images in batch"""
        try:
            # Decode all base64 images
            images = []
            images: List[Image.Image] = []
            for img_base64 in images_base64:
                images += [self.decode_base64_to_image(img_base64)]
            
            # Process the images
            batch_images = self.processor.process_images(images).to(self.model.device)
            
            # Generate embeddings
            with torch.no_grad():
                embeddings = self.model(**batch_images)
                
            # Convert to list
            embeddings_np = embeddings.cpu().to(torch.float32).detach().numpy()
            return embeddings_np
        except Exception as e:
            raise ValueError(f"Batch image embedding generation failed: {str(e)}")
    
    @modal.method() 
    async def embed_query(self, query: str) -> np.ndarray: 
        import torch
        """Generate embedding for a single query"""
        try:
            # Process the query
            batch_query = self.processor.process_queries([query]).to(self.model.device)
            
            # Generate embedding
            with torch.no_grad():
                embedding = self.model(**batch_query)
                
            # Convert to list
            embedding_np = embedding.cpu().to(torch.float32).detach().numpy()
            return embedding_np[0]
        except Exception as e:
            raise ValueError(f"Query embedding generation failed: {str(e)}")
    
    @modal.method()
    async def batch_embed_queries(self, queries: List[str]) -> np.ndarray: 
        import torch
        """Generate embeddings for multiple queries in batch"""
        try:
            # Process the queries
            batch_queries = self.processor.process_queries(queries).to(self.model.device)
            
            # Generate embeddings
            with torch.no_grad():
                embeddings = self.model(**batch_queries)
                
            # Convert to list
            embeddings_np = embeddings.cpu().to(torch.float32).detach().numpy()
            return embeddings_np
        except Exception as e:
            raise ValueError(f"Batch query embedding generation failed: {str(e)}")
    
    @modal.method()
    async def score(self, queries: List[str], images_base64: List[str]) -> np.ndarray:
        import torch
        """Score queries against images"""
        try:
            # Decode all base64 images
            images = []
            for img_base64 in images_base64:
                img = self.decode_base64_to_image(img_base64)
                images.append(img)
            
            # Process inputs
            batch_images = self.processor.process_images(images).to(self.model.device)
            batch_queries = self.processor.process_queries(queries).to(self.model.device)
            
            # Forward pass
            with torch.no_grad():
                image_embeddings = self.model(**batch_images)
                query_embeddings = self.model(**batch_queries)
            
            # Calculate scores
            scores = self.processor.score_multi_vector(query_embeddings, image_embeddings)
            
            return scores.cpu().to(torch.float32).detach().numpy()
        except Exception as e:
            raise ValueError(f"Scoring failed: {str(e)}") 