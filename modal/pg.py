from vechord.registry import VechordRegistry
from vechord.spec import PrimaryKeyAutoIncrease, Table, Vector, MultiVectorIndex, IndexColumn
from typing import List, Optional, Annotated
from pathlib import Path
from typing import Iterator
from vechord.evaluate import BaseEvaluator
import datasets
import msgspec
import time

MultiVector = List[Vector[128]]

lists = 2500
class Image(Table, kw_only=True):
    uid: Optional[PrimaryKeyAutoIncrease] = None
    image_embedding: Annotated[MultiVector, MultiVectorIndex(lists=lists)] 
    query_embedding: Annotated[MultiVector, MultiVectorIndex(lists=lists)]
    query: str = None
    dataset: Optional[str] = None
    dataset_id: Optional[int] = None

vr = VechordRegistry("colpali", "postgresql://postgres:postgres@127.0.0.1:5440/")
vr.register([Image])

@vr.inject(output=Image)
def load_image(path: str) -> Iterator[Image]:
    import numpy as np
    for dataset_path in Path(path).iterdir():
        dataset_name = '/'.join(str(dataset_path).split('/')[-2:])
        dataset = datasets.load_from_disk(dataset_path=dataset_path)
        for i,item in enumerate(dataset):
            image_embedding = np.frombuffer(item['image_embedding'], dtype=np.float32).reshape(item['image_embedding_shape'])
            vector_dim = image_embedding.shape[1] if len(image_embedding.shape) > 1 else len(image_embedding[0])
            image_vectors = [Vector[vector_dim](embedding) for embedding in image_embedding]

            query_embedding = np.frombuffer(item['query_embedding'], dtype=np.float32).reshape(item['query_embedding_shape'])
            vector_dim = query_embedding.shape[1] if len(query_embedding.shape) > 1 else len(query_embedding[0])
            query_vectors = [Vector[vector_dim](embedding) for embedding in query_embedding]
            yield Image(
                image_embedding=image_vectors, query_embedding=query_vectors,query=item['query'], dataset=dataset_name, dataset_id=i
            )

class Evaluation(msgspec.Struct):
    map: float
    ndcg: float
    recall: float

TOP_K = 10
def evaluate(queries: list[Image], probes: int, maxsim_refine: int) -> list[Evaluation]:
    result  = []
    for query in queries:
        vector = query.query_embedding
        docs: list[Image] = vr.search_by_multivec(
            Image, vector, topk=TOP_K, probe=probes, maxsim_refine=maxsim_refine
        )
        score = BaseEvaluator.evaluate_one(query.uid, [doc.uid for doc in docs])
        result.append(Evaluation(
            map=score.get("map"),
            ndcg=score.get("ndcg"),
            recall=score.get(f"recall_{TOP_K}"),
        ))
    return result

if __name__ == "__main__":
    load_image("/home/xieyuandong/pgvecto.rs-cloud/poc/hybrid-search/vectorchord-colpali/modal/vidore")
    queries: list[Image] = vr.select_by(Image.partial_init(dataset="vidore/arxivqa_test_subsampled"), limit=100)
    # Measure latency and throughput for evaluation
    start_time = time.time()
    count = 2229
    lists = 2500
    probes = 16
    maxsim_refine = 50
    epsilon = 0.1
    maxsim_threshold = 2000
    with vr.client.get_cursor() as cursor:
        cursor.execute(f"SET vchordrq.maxsim_threshold={maxsim_threshold}")
        cursor.execute(f"SET vchordrq.epsilon={epsilon}")

    res: list[Evaluation] = evaluate(queries, probes=probes, maxsim_refine=maxsim_refine)
    end_time = time.time()

    # Calculate metrics
    total_time = end_time - start_time
    avg_latency = total_time / len(queries)  # Average latency per query (seconds)
    throughput = len(queries) / total_time   # Throughput (queries per second)

    print("probes", probes)
    print("maxsim_threshold", maxsim_threshold)
    print("epsilon", epsilon)
    print("ndcg@10", sum(r.ndcg for r in res) / len(res))
    print("recall@10", sum(r.recall for r in res) / len(res))
    print(f"Total execution time: {total_time:.4f} seconds")
    print(f"Average latency: {avg_latency*1000:.4f} ms/query")
    print(f"Throughput: {throughput:.2f} queries/second\n")
    # vr.clear_storage()