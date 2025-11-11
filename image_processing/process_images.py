import os
import ray
from huggingface_hub import HfFileSystem
from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor
from PIL import Image
from io import BytesIO
import urllib3
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging



# ============================================================================
# SCALABILITY CONFIGURATION FOR 2B+ IMAGES
# ============================================================================
# num_images = 100
num_model_replicas = 32
tensor_parallelism = 1
max_concurrent_downloads = 10 

from datetime import datetime, timezone

timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

output_path = f"/mnt/shared_storage/process_images_output/{timestamp}"


def image_download(batch):
    """Get the url from the batch and download the image,
    if successful, return the image bytes, otherwise return None"""
    import httpx  # Much faster than urllib3
    
    # httpx with HTTP/2 support for multiplexing
    client = httpx.Client(
        http2=True,
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=50),
        timeout=httpx.Timeout(5.0, connect=3.0),
        follow_redirects=True,
    )
    
    def download_single(url):
        if not url or not isinstance(url, str):
            return None
        url_lower = url.lower().strip()
        if not (url_lower.startswith('http://') or url_lower.startswith('https://')):
            return None
        try:
            response = client.get(url)
            if response.status_code == 200 and response.content:
                return response.content
            return None
        except Exception:
            return None
    
    urls = batch["url"]
    
    # Increase thread pool to match batch processing
    with ThreadPoolExecutor(max_workers=50) as executor:
        results = list(executor.map(download_single, urls))
    
    batch["bytes"] = results
    return batch

def process_image_bytes(batch):
    """Process image bytes"""
    image_bytes_list = batch["bytes"]
    
    def process_single_image(image_bytes):
        """Process a single image, return processed bytes or None on failure"""
        if image_bytes is None:
            return None
        try:
            img = Image.open(BytesIO(image_bytes))
            img.load()  # Force full image loading to detect truncation
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = img.resize((128, 128), Image.Resampling.LANCZOS)
            output_buffer = BytesIO()
            img.save(output_buffer, format="JPEG", quality=95)
            return output_buffer.getvalue()
        except Exception:
            return None
    
    # Process each image in the batch
    with ThreadPoolExecutor(max_workers=50) as executor:
        results = list(executor.map(process_single_image, image_bytes_list))
    batch["bytes"] = results
    return batch


vision_processor_config = vLLMEngineProcessorConfig(
    model_source="Qwen/Qwen2.5-VL-3B-Instruct",
    engine_kwargs=dict(
        tensor_parallel_size=tensor_parallelism,
        pipeline_parallel_size=1,
        max_model_len=32768,
        enable_chunked_prefill=True,
        max_num_batched_tokens=2048,
    ),
    # Override Ray's runtime env to include the Hugging Face token. Ray Data uses Ray under the hood to orchestrate the inference pipeline.
    runtime_env=dict(
        env_vars=dict(
            VLLM_USE_V1="1",
            VLLM_DISABLE_COMPILE_CACHE="1",
        ),
    ),
    batch_size=8,  # Reduced from 16 to lower memory usage
    max_concurrent_batches=16,  # Increased to saturate vLLM engine (8 * 16 = 128)
    accelerator_type="A10G",
    concurrency=num_model_replicas,
    has_image=True,
)


def vision_preprocess(row: dict) -> dict:
    # Keep image data as base64 string for Arrow serialization
    # The vLLM engine will handle the conversion internally
    image_bytes = row["bytes"]
    return dict(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": Image.open(BytesIO(image_bytes)),
                    },
                ],
            },
        ],
        sampling_params=dict(
            temperature=0.3,
            max_tokens=150,
            detokenize=False,
        ),
    )


def vision_postprocess(row: dict) -> dict:
    row.pop("bytes")
    return row


vision_processor = build_llm_processor(
    vision_processor_config,
    preprocess=vision_preprocess,
    postprocess=vision_postprocess,
)


num_cpu = 512
tasks_per_cpu = 1
concurrency = num_cpu * tasks_per_cpu
ctx = ray.data.DataContext.get_current()
target_block_size_mb = 128
ctx.target_max_block_size = target_block_size_mb * 1024 * 1024
ctx.use_push_based_shuffle = False


# Data pipeline with scalability optimizations
dataset = (
    ray.data.read_parquet(
        "hf://datasets/laion/relaion2B-en-research-safe/",
        file_extensions=["parquet"],
        columns=["url"],
        filesystem=HfFileSystem(token=os.environ["HF_TOKEN"]),
        concurrency=concurrency,
        num_cpus=2,
        memory=int(4 * 1024**3),
    )
    .map_batches(image_download, batch_size=50, num_cpus=0.5, concurrency=1024)
    .drop_columns(["url"])
    .map_batches(
        process_image_bytes,
        batch_size=50,
        num_cpus=1,
    )
    .filter(lambda row: row["bytes"] is not None)
) 


# Apply vision processing with scaled replicas
# Note: image_base64 column is dropped in vision_postprocess to avoid Arrow serialization issues
dataset = vision_processor(dataset)

# Write with optimizations for throughput and fault tolerance
dataset.write_parquet(
    output_path,
    max_rows_per_file=100000,  # ~100K rows per file for manageable file sizes
)
