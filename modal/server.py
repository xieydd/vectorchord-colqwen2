from fastapi import FastAPI, HTTPException,Response
from fastapi.responses import JSONResponse
from typing import List
from colpali import ColPaliModel
from pydantic import BaseModel
import modal
import subprocess
import socket
import logging
from msgspec import Struct
import msgspec
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GPU_CONFIG = "H100"
GPU_CONCURRENCY = 1
# Define FastAPI app
app = FastAPI(title="ColQwen2 API", description="API for ColQwen2 embedding model")

# Define Modal APP
modal_app = modal.App("colqwen2-embedding-service")

class ImageEmbeddingRequest(BaseModel):
    image: str

class QueryEmbeddingRequest(BaseModel):
    query: str

class EmbeddingResponse(Struct):
    embedding: bytes
    shape: tuple[int, int]
    dtype: str

class ImageBatchEmbeddingRequest(BaseModel):
    images: List[str]

class QueryBatchEmbeddingRequest(BaseModel):
    queries: List[str]

class BatchEmbeddingResponse(Struct):
    embeddings: bytes
    shape: tuple[int, int, int]
    dtype: str

class ScoringRequest(BaseModel):
    queries: List[str]
    images: List[str]

class ScoringResponse(Struct):
    scores: bytes
    shape: tuple[int, int]
    dtype: str

def colpali_app():
    # Create Modal class client
    model = ColPaliModel()
    encoder = msgspec.msgpack.Encoder()
    
    @app.get("/health")
    async def health_check():
        try:
            return {"status": "ok"}
        except Exception as e:
            return JSONResponse(status_code=503, content={"status": "Service Unavailable", "error": str(e)})
    
    @app.post("/embed_image")
    async def embed_image(request: ImageEmbeddingRequest):
        try:
            embedding = await model.embed_image(request.image)
            response = EmbeddingResponse(
                embedding=embedding.tobytes(),
                shape=embedding.shape,
                dtype=str(embedding.dtype)
            )
            response_bytes = encoder.encode(response)
            return Response(content=response_bytes, media_type="application/octet-stream")
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Image embedding generation failed: {str(e)}")
    
    @app.post("/batch_embed_images")
    async def batch_embed_images(request: ImageBatchEmbeddingRequest):
        try:
            embeddings = await model.batch_embed_images(request.images)
            response = BatchEmbeddingResponse(
                embeddings=embeddings.tobytes(),
                shape=embeddings.shape,
                dtype=str(embeddings[0].dtype)
            )
            response_bytes = encoder.encode(response)
            return Response(content=response_bytes, media_type="application/octet-stream")
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Batch image embedding generation failed: {str(e)}")
    
    @app.post("/embed_query")
    async def embed_query(request: QueryEmbeddingRequest):
        try:
            embedding = await model.embed_query(request.query)
            response = EmbeddingResponse(
                embedding=embedding.tobytes(),
                shape=embedding.shape,
                dtype=str(embedding.dtype)
            ) 
            response_bytes = encoder.encode(response)
            return Response(content=response_bytes, media_type="application/octet-stream")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Query embedding generation failed: {str(e)}")
    
    @app.post("/batch_embed_queries")
    async def batch_embed_queries(request: QueryBatchEmbeddingRequest):
        try:
            embeddings = await model.batch_embed_queries(request.queries)
            response = BatchEmbeddingResponse(
                embeddings=embeddings.tobytes(),
                shape=embeddings.shape,
                dtype=str(embeddings[0].dtype)
            )
            response_bytes = encoder.encode(response)
            return Response(content=response_bytes, media_type="application/octet-stream")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Batch query embedding generation failed: {str(e)}")
    
    @app.post("/score")
    async def score(request: ScoringRequest):
        try:
            scores_list = await model.score(request.queries, request.images)
            response = ScoringResponse(
                scores=scores_list.tobytes(),
                shape=scores_list[0].shape,
                dtype=str(scores_list[0].dtype)
            )
            response_bytes = encoder.encode(response)
            return Response(content=response_bytes, media_type="application/octet-stream")
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")
    
    return app

def spawn_server() -> subprocess.Popen:
    """Start FastAPI server as a subprocess"""
    process = subprocess.Popen(["uvicorn", "server:colpali_app", "--host", "127.0.0.1", "--port", "8000"])
    # Poll until 127.0.0.1:8000 accepts connections
    while True:
        try:
            socket.create_connection(("127.0.0.1", 8000), timeout=1).close()
            print("Webserver ready!")
            return process
        except (socket.timeout, ConnectionRefusedError):
            # Check if webserver process has exited
            retcode = process.poll()
            if retcode is not None:
                raise RuntimeError(f"FastAPI server exited unexpectedly with code {retcode}")


# Define Modal image
server_image = modal.Image.debian_slim().pip_install(
    "fastapi",
    "uvicorn",
    "pydantic",
    "pillow",
    "colpali_engine",
    "httpx",
    "msgspec"
)

MODEL_CACHE_VOLUME = modal.Volume.from_name(
    "colpali-model-cache", create_if_missing=True
)

MODEL_DIR = "/model"

@modal_app.cls(
    gpu=GPU_CONFIG,
    timeout=600,
    image=server_image,
    max_containers=GPU_CONCURRENCY,
    allow_concurrent_inputs=True,
    retries=3,
    volumes={
        MODEL_DIR: MODEL_CACHE_VOLUME ,
    },
)
class ColPaliServer:
    @modal.enter()
    def open_connection(self):
        # If the process is running for a long time, the client does not seem to close the connections, results in a pool timeout
        from httpx import AsyncClient
        self.process = spawn_server()
        self.client = AsyncClient(base_url="http://127.0.0.1:8000", timeout=30)
    
    @modal.exit()
    def terminate_connection(self):
        """Terminate server connection"""
        self.process.terminate()
    
    @modal.method()
    def health_check(self):
        """Health check endpoint"""
        return {"status": "ok"}

    async def _embed_images(self, images: List[str]) -> np.ndarray:
        import numpy as np
        import httpx
        try:
            data = ImageBatchEmbeddingRequest(images=images) 
            res = await self.client.post(
                "http://127.0.0.1:8000/batch_embed_images",
                json=data.model_dump(),
            )
            res.raise_for_status()
            decoder = msgspec.msgpack.Decoder(BatchEmbeddingResponse)
            response_obj = decoder.decode(res.content)
            numpy_array = np.frombuffer(response_obj.embeddings, dtype=response_obj.dtype).reshape(response_obj.shape)
            return numpy_array
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error occurred: {str(e)}")
            logger.error(f"Response content: {e.response.content}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise

    @modal.method()
    async def embed_images(self, images: List[str]) -> np.ndarray:
        # Ensure images is a list type
        if not isinstance(images, list):
            raise ValueError("Images must be a list")
            
        if not images:
            raise ValueError("Empty images list")
            
        embeddings = await self._embed_images(images)
        return embeddings

    async def _embed_queries(self, queries: List[str]) -> np.ndarray:
        import numpy as np
        import httpx
        try:
            data = QueryBatchEmbeddingRequest(queries=queries)
            res = await self.client.post(
                "http://127.0.0.1:8000/batch_embed_queries",
                json=data.model_dump(),
            )
            res.raise_for_status()
            decoder = msgspec.msgpack.Decoder(BatchEmbeddingResponse)
            response_obj = decoder.decode(res.content)
            numpy_array = np.frombuffer(response_obj.embeddings, dtype=response_obj.dtype).reshape(response_obj.shape)
            return numpy_array
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error occurred: {str(e)}")
            logger.error(f"Response content: {e.response.content}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise
    
    @modal.method()
    async def embed_queries(self, queries: List[str]) -> np.ndarray:
        # Ensure queries is a list type
        if not isinstance(queries, list):
            raise ValueError("Queries must be a list")
            
        if not queries:
            raise ValueError("Empty queries list")
            
        embeddings = await self._embed_queries(queries)
        return embeddings