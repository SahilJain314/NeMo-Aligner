# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
#  python serve_vllm_inference_client.py trainer.grpo.inference_backend.config.vllm.enable=True trainer.grpo.inference_backend.config.vllm.port=$VLLM_PORT trainer.grpo.inference_backend.config.vllm.ip=$VLLM_IP
from nemo_aligner.experimental.grpo.inference.vllm.vllm_client import VLLMClient
from nemo.core.config import hydra_runner
from transformers import AutoTokenizer
import torch
import os
import requests
from datetime import timedelta
from megatron.core import parallel_state as mcore_parallel_state

from torch._C._distributed_c10d import PrefixStore
from torch.distributed import rendezvous

CHECKPOINT_PATH="/opt/checkpoints"


def world_size():
    """Lazily grab device count"""
    return torch.cuda.device_count()

def rank():
    """Lazily grab rank"""
    return int(os.environ["LOCAL_RANK"])

def initialize_distributed():
    if not torch.distributed.is_initialized():
        torch.cuda.set_device(rank() % torch.cuda.device_count())
        init_method = "tcp://"
        master_ip = os.getenv("MASTER_ADDR", "localhost")
        master_port = os.getenv("MASTER_PORT", "6000")
        init_method += master_ip + ":" + master_port
        rendezvous_iterator = rendezvous(init_method, rank(), world_size(), timeout=timedelta(minutes=1))
        store, _, _ = next(rendezvous_iterator)
        store.set_timeout(timedelta(minutes=1))

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore("default_pg", store)

        torch.distributed.init_process_group(
            backend="nccl", world_size=world_size(), rank=rank(), store=store,
        )

        torch.distributed.barrier()


MODEL = 'meta-llama/Llama-3.1-8B-Instruct'

@hydra_runner(config_path="conf", config_name="gpt_grpo")
def main(cfg) -> None:
    initialize_distributed()
    mcore_parallel_state.initialize_model_parallel(
                pipeline_model_parallel_size=1,
                tensor_model_parallel_size=1,
                context_parallel_size=1,
            )

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.eos_id = 0
    tokenizer.pad_token = tokenizer.eos_token
    inference_backend = VLLMClient(
        cfg.trainer.grpo.inference_backend.config.vllm,
        tokenizer=tokenizer,
        checkpoint_path=CHECKPOINT_PATH,
    )
    input_texts_list = [
        "Who is the greatest basketball player of all time?",
        "def fibonacci",
    ]
    tokens = tokenizer(input_texts_list, padding=True, return_tensors="pt")
    input_ids = tokens['input_ids']

    inference_batch = {
        "text": input_ids,
        "length": (input_ids != tokenizer.pad_token_id).sum(dim=1)
    }

    prompt_tokens = inference_batch["text"].cuda(non_blocking=True)
    prompt_lengths = inference_batch["length"].cuda(non_blocking=True)
    inputs = (prompt_tokens, prompt_lengths)


    url = f"http://{cfg.trainer.grpo.inference_backend.config.vllm.ip}:{cfg.trainer.grpo.inference_backend.config.vllm.port}"
    requests.post(f'{url}/start',
        json={
            "checkpoint_path": CHECKPOINT_PATH,
            "tp": 4,  # hardcode for testing now
            "tp_src_gpu_idx": 0,  # hardcode for testing now
        }
    )
    inference_backend.start = True

    actor_output = inference_backend.generate(inputs, use_greedy=True)
    response_tokens = actor_output["response_tokens"]
    ref_response_trt_lps = actor_output["response_logprobs_trt"]
    decoded_texts = tokenizer.batch_decode(response_tokens, skip_special_tokens=True)
    print(f"right: {decoded_texts}", flush=True)

    # reset all parameter to zeros
    requests.post(f'{url}/refit_zero',
        json={
            'checkpoint_path': CHECKPOINT_PATH,
        }
    )
    actor_output = inference_backend.generate(inputs, use_greedy=True)
    response_tokens = actor_output["response_tokens"]
    decoded_texts = tokenizer.batch_decode(response_tokens, skip_special_tokens=True)
    print(f"random: {decoded_texts}", flush=True)

    # refit
    inference_backend.refit('/opt/checkpoints/')
    actor_output = inference_backend.generate(inputs, use_greedy=True)
    response_tokens = actor_output["response_tokens"]
    response_trt_lps = actor_output["response_logprobs_trt"]
    decoded_texts = tokenizer.batch_decode(response_tokens, skip_special_tokens=True)
    print(f"right: {decoded_texts}", flush=True)

    assert torch.allclose(ref_response_trt_lps, response_trt_lps, atol=1e-8, rtol=1e-5)


if __name__ == "__main__":
    main()