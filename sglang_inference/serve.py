"""Ray Serve deployment for SGLang inference.

This deployment uses Ray Serve's placement_group_bundles to reserve GPUs
across multiple nodes for tensor-parallel inference with SGLang.

The Engine is created in a dedicated background thread to avoid event loop
conflicts with Ray Serve's async actor. A warmup request initializes NCCL
pipeline-parallel groups before serving traffic.

Based on the Ray Serve LLM + SGLang integration pattern from:
https://github.com/ray-project/ray/pull/58366
"""

import asyncio
import os
import signal
import logging
import threading

from fastapi import FastAPI
from ray import serve

# Configuration from environment (same defaults as driver_offline.py)
MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen3.5-27B")
TP_SIZE = int(os.environ.get("TP_SIZE", "4"))
PP_SIZE = int(os.environ.get("PP_SIZE", "2"))
NUM_NODES = int(os.environ.get("NUM_NODES", "2"))

gpus_per_node = (TP_SIZE * PP_SIZE) // NUM_NODES

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 4,
        "target_ongoing_requests": 4,
    },
    ray_actor_options={
        "num_cpus": 1,
        "num_gpus": 0,
    },
    # Reserve resources across multiple nodes for tensor parallelism.
    # Each bundle reserves GPUs on one node.
    placement_group_bundles=[{"CPU": 1, "GPU": gpus_per_node}] * NUM_NODES,
)
@serve.ingress(app)
class SGLangDeployment:
    def __init__(self):
        init_done = threading.Event()
        init_error = [None]

        def _engine_thread():
            try:
                # Import sglang inside the actor because it initializes CUDA
                # and cannot be imported on the CPU-only head node where the
                # Serve controller runs.
                from sglang.srt.ray.engine import RayEngine as Engine

                # Monkey-patch signal.signal to avoid "signal only works in
                # main thread" error during Engine init.
                original_signal = signal.signal
                signal.signal = lambda sig, action: signal.SIG_DFL
                try:
                    logger.info(f"Initializing engine with model path: {MODEL_PATH}")
                    self.engine = Engine(
                        model_path=MODEL_PATH,
                        tp_size=TP_SIZE,
                        pp_size=PP_SIZE,
                        nnodes=NUM_NODES,
                    )
                finally:
                    signal.signal = original_signal

                # Warmup — initializes NCCL pipeline-parallel groups so the
                # first real request doesn't hang.
                logger.info("Running warmup request...")
                self.engine.generate(
                    prompt="warmup",
                    sampling_params={"max_new_tokens": 1},
                )
                logger.info("Warmup complete, engine ready.")

                init_done.set()

                # Keep the Engine's event loop running so its internal async
                # tasks (TokenizerManager handle_loop, ZMQ polling) stay active.
                self.engine.loop.run_forever()
            except Exception as e:
                init_error[0] = e
                init_done.set()

        self._loop_thread = threading.Thread(target=_engine_thread, daemon=True)
        self._loop_thread.start()
        init_done.wait()

        if init_error[0]:
            raise init_error[0]

    @app.post("/")
    async def generate(self, request: dict) -> dict:
        text = request.get("text", "")
        sampling_params = request.get("sampling_params", {"max_new_tokens": 64})

        # Bridge Ray Serve's event loop to the Engine's background loop.
        future = asyncio.run_coroutine_threadsafe(
            self.engine.async_generate(
                prompt=text, sampling_params=sampling_params,
            ),
            self.engine.loop,
        )
        result = await asyncio.wrap_future(future)
        return {"text": result["text"]}


app_deploy = SGLangDeployment.bind()
