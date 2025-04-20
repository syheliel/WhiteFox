import click
from typing import TypedDict
from loguru import logger
from src.db.factory import EmbeddingFactory, EmbeddingType, ChromaVectorDB
from src.db.factory import BaseEmbedding
import tqdm
from typing import List
import concurrent.futures
class SpecData(TypedDict):
    summary: str
    python_code: str
    api: List[str]

LEVEL1_PROMPTS = [
    "the description of <API_NAME>",
    "the parameter and input/output tensor of <API_NAME>",
    "the usage of calling <API_NAME>",
]

LEVEL2_PROMPTS = [
    "<API_NAME>'s __DOC__",
    "the parameter and input/output tensor of <API_NAME> defined by shpinx format, especially text with :math: and :attr:",
    "the usage of calling <API_NAME>",
]

class Result(TypedDict):
    prompt_1: List[str]
    prompt_2: List[str]
    prompt_3: List[str]

SCORE = [[0, 0, 0], [0, 0, 0]]
def get_api_docs(apis:List[str],embedding:BaseEmbedding, vector_db:ChromaVectorDB):
    for api in apis:

        for i, prompt in enumerate([LEVEL1_PROMPTS, LEVEL2_PROMPTS]):
            p1 = prompt[0].replace("<API_NAME>", api)
            p2 = prompt[1].replace("<API_NAME>", api)
            p3 = prompt[2].replace("<API_NAME>", api)
            query_embedding = embedding.embed_query(p1)
            docs:List[str] = vector_db.query_by_emb(query_embedding, n_results=1)["documents"][0] # type: ignore
            for doc in docs: # type: ignore
                print(doc)
                if api in doc:
                    SCORE[i][0] += 1
            query_embedding = embedding.embed_query(p2)
            docs:List[str] = vector_db.query_by_emb(query_embedding, n_results=1)["documents"][0] # type: ignore
            for doc in docs: # type: ignore
                if api in doc:
                    SCORE[i][1] += 1
            query_embedding = embedding.embed_query(p3)
            docs:List[str] = vector_db.query_by_emb(query_embedding, n_results=1)["documents"][0] # type: ignore
            for doc in docs: # type: ignore
                print(doc)
                if api in doc:
                    SCORE[i][2] += 1

@click.command()
@click.option("--device", type=str, default="cpu", help="Device to use for generate prompt")
@click.option("--max-workers", type=int, default=2, help="Maximum number of parallel workers")
def main(device: str, max_workers: int):
    embedding = EmbeddingFactory.create_embedding(EmbeddingType.HUGGINGFACE, device=device)
    vector_db = ChromaVectorDB(embedding)
    # print(vector_db.collection.get())
    # return
    
    with open("./optim-0/oplist.txt", "r") as f:
        op_list:List[str] = f.read().splitlines()
    
    logger.info(f"Starting to process {len(op_list)} operations with {max_workers} workers")
    
    # Use ThreadPoolExecutor for I/O-bound operations
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all operations to the executor
        future_to_op = {executor.submit(get_api_docs, [op], embedding, vector_db): op for op in op_list}
        
        # Process the results as they complete
        completed = 0
        for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_op), total=len(op_list), desc="Processing operations"):
            op = future_to_op[future]
            completed += 1
            try:
                result = future.result()
                if result:
                    logger.info(f"Completed processing {result} ({completed}/{len(op_list)})")
            except Exception as exc:
                logger.error(f"{op} generated an exception: {exc}")
                raise exc
    
    logger.info(f"✅ All operations processed successfully. Results saved in {output_dir}")
    logger.info(f"SCORE: {SCORE}")
if __name__ == "__main__":
    main()