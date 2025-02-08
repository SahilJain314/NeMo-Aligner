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

import argparse
import os
import contextlib
from flask import Flask, request, jsonify
import threading
import time
import random
import multiprocessing as mp
from tqdm.auto import tqdm

app = Flask(__name__)

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

# Global variable to hold the inference server instance
#inference_server = None
in_q = mp.Queue()
out_q = mp.Queue()
inference_process = None


def worker_process(in_queue, out_queue, load_path, tp):
    import torch
    import glob
    import gc
    from vllm import LLM, SamplingParams
    from vllm.utils import get_ip, get_open_port
    from vllm.worker.worker import Worker

    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
    import time
    from vllm.model_executor.model_loader.weight_utils import pt_weights_iterator, safetensors_weights_iterator

    class MyWorker(Worker):
        def clean_cuda_memory(self):
            torch.cuda.empty_cache()
            # Force garbage collection
            import gc
            gc.collect()

        def reset(self):
            self.model_runner.builder.reset_cached_inter_data()
            self.clean_cuda_memory()

        def update_full_weights(self, path, zero_init=False):
            """Update weights with synchronization across all nodes"""
            # Update local model weights
            bin_files = glob.glob(os.path.join(path, "*.bin"))
            safetensor_files = glob.glob(os.path.join(path, "*.safetensors"))

            # Check if both formats exist
            if bin_files and safetensor_files:
                raise ValueError(
                    f"Found both .bin and .safetensors files in {path}. "
                    f"Only one format type should exist in the folder.\n"
                    f"bin files: {bin_files}\n"
                    f"safetensor files: {safetensor_files}"
                )

            if bin_files:
                weight_iterator = pt_weights_iterator(bin_files)
            elif safetensor_files:
                weight_iterator = safetensors_weights_iterator(safetensor_files)
            else:
                raise ValueError(f"No Checkpoint files(*.bin or *.safetensors) found under {path=}")
            for name, weight in weight_iterator:
                if zero_init:
                    weight.zero_()
                self.model_runner.model.load_weights(weights=[(name, weight)])
                del weight

    class VLLMInferenceServer:
        """
        This class simulates a vLLM inference server.
        Replace the simulation with actual vLLM startup,
        shutdown, and inference logic as needed.
        """
        def __init__(self):
            self.running = False
            self.llm = None

        def start(self, path, tp):
            start = time.time()

            # TODO: use low gpu_memory_utilization=.3 increase it back to .62 once we optimize gpu memory efficiency
            self.llm = LLM(model=path, device='cuda', worker_cls=MyWorker, tensor_parallel_size=tp, generation_config='auto', enforce_eager=True, gpu_memory_utilization=.3, max_num_batched_tokens=32768)
            self.running = True
            elapsed_time = time.time() - start
            print(f"vLLM Inference Server started. ({elapsed_time:.2f} seconds elapsed)")

        def refit(self, path, zero_init: bool = False):
            start = time.time()
            if not isinstance(path, str):
                raise ValueError("a string path is needed to pass refit")

            # Total TP workers i.e. 1 driver_worker along with TP - 1 spawn workers

            # reset
            self.llm.llm_engine.reset_prefix_cache()
            for worker in self.llm.llm_engine.model_executor.workers:
                worker.execute_method("reset")
            self.llm.llm_engine.model_executor.driver_worker.worker.reset()

            # load weights
            for worker in self.llm.llm_engine.model_executor.workers:
                worker.execute_method("update_full_weights", path, zero_init=zero_init)
            self.llm.llm_engine.model_executor.driver_worker.worker.update_full_weights(path, zero_init=zero_init)

            elapsed_time = time.time() - start
            print(f"vLLM Refit ({elapsed_time:.2f} seconds elapsed)")
            return True


        def shutdown(self):
            destroy_model_parallel()
            destroy_distributed_environment()
            del self.llm.llm_engine.model_executor
            del self.llm
            with contextlib.suppress(AssertionError):
                torch.distributed.destroy_process_group()
            gc.collect()
            torch.cuda.empty_cache()
            self.llm = None
            self.running = False
            print("vLLM Inference Server shutdown.")

        def generate(self, batch_tokens, greedy=False):
            """
            For each sequence in batch_tokens, we generate:
              - a new sequence of output tokens
              - a list of logprobs (one per token in the generated sequence)
            """
            # greedy decoding applied if temperature is below a threshold in vllm
            temperature = 0. if greedy else 1.
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=1.0,
                top_k=-1,
                logprobs=0,
                max_tokens=2048,
                #ignore_eos=True,
            )

            # Generate texts from the prompts. The output is a list of RequestOutput objects
            # that contain the prompt, generated text, and other information.
            prompt_tokens = [TokensPrompt(prompt_token_ids=tok_seq) for tok_seq in batch_tokens]
            outputs = self.llm.generate(prompt_tokens, sampling_params, use_tqdm=True)
            logprobs = []
            out_tokens = []
            for output in outputs:
                lps = [next(iter(l.items()))[1].logprob for l in output.outputs[0].logprobs]
                out_toks = [next(iter(l.items()))[0] for l in output.outputs[0].logprobs]
                logprobs.append(lps)
                out_tokens.append(out_toks)

            return out_tokens, logprobs

    server = VLLMInferenceServer()
    server.start(load_path, tp)

    while True:
        command, args = in_queue.get()
        if command == "shutdown":
            server.shutdown()
            return
        elif command == "generate":
            batch_tokens = args[0]
            out_queue.put(server.generate(batch_tokens))
        elif command == "generate_greedy":
            batch_tokens = args[0]
            out_queue.put(server.generate(batch_tokens, greedy=True))
        elif command == "refit":
            path = args[0]
            out_queue.put(server.refit(path))
        elif command == "refit_zero":
            path = args[0]
            out_queue.put(server.refit(path, zero_init=True))
        else:
            raise NotImplementedError("Unknown Command")


def refit_server_fn(command_name):
    global in_q
    global out_q
    global inference_process
    if inference_process is None:
        return jsonify({"status": "no running vllm"}), 200

    data = request.get_json()
    in_q.put((command_name, (data['checkpoint_path'], )))
    _ = out_q.get()
    return jsonify({"status": "refit success"}), 200


@app.route('/refit', methods=['POST'])
def refit_server():
    return refit_server_fn('refit')


@app.route('/refit_zero', methods=['POST'])
def refit_zero_server():
    return refit_server_fn('refit_zero')


@app.route('/start', methods=['POST'])
def start_server():
    """
    Starts the vLLM inference server. Returns only when the server is ready.
    """
    global inference_process
    if inference_process is not None:
        return jsonify({"status": "already running"}), 200

    data = request.get_json()
    # set CUDA_VISIBLE_DEVICES to the TP group (str)
    os.environ["CUDA_VISIBLE_DEVICES"]=",".join(map(str, list(range(data["tp_src_gpu_idx"], data["tp_src_gpu_idx"]+data["tp"]))))
    inference_process = mp.Process(target=worker_process, args=(in_q, out_q, data["checkpoint_path"], data["tp"]))
    inference_process.start()
    return jsonify({"status": "started"}), 200


@app.route('/shutdown', methods=['POST'])
def shutdown_server():
    """
    Shuts down the vLLM inference server.
    """
    #global inference_server
    global inference_process
    global in_q
    #if inference_server is None or not inference_server.running:
    #    return jsonify({"status": "inference server not running"}), 400

    in_q.put(("shutdown", None))
    inference_process.join()
    inference_process = None
    #inference_server.shutdown()
    return jsonify({"status": "shutdown initiated"}), 200


def generate_fn(command_name):
    global in_q
    global out_q
    global inference_process
    if inference_process is None:
        return jsonify({"error": "inference server not running"}), 400

    import time
    start_time = time.time()
    data = request.get_json()
    end_time = time.time()
    print(f"Request get_json time: {end_time - start_time} seconds")
    if not isinstance(data, list):
        return jsonify({"error": "Expected a list of lists of tokens"}), 400

    # Validate that each element in the list is itself a list
    for i, item in enumerate(data):
        if not isinstance(item, list):
            return jsonify({"error": f"Element at index {i} is not a list"}), 400

    in_q.put((command_name, (data,)))
    generations, logprobs = out_q.get()
    response = {
        "response_tokens": generations,
        "response_logprobs": logprobs
    }
    import time
    start_time = time.time()
    response = jsonify(response), 200
    end_time = time.time()
    print(f"Response jsonify time: {end_time - start_time} seconds")
    return response


@app.route('/generate', methods=['POST'])
def generate():
    """
    Performs inference on a batch of token lists.
    Expects a JSON payload that is a list of lists of tokens.
    Returns:
      - generations: list of generated token sequences for each batch element
      - logprobs: list of token-wise log-probabilities for each generation
    """
    return generate_fn("generate")


@app.route('/generate_greedy', methods=['POST'])
def generate_greedy():
    """
    Performs inference on a batch of token lists.
    Expects a JSON payload that is a list of lists of tokens.
    Returns:
      - generations: list of generated token sequences for each batch element
      - logprobs: list of token-wise log-probabilities for each generation
    """
    return generate_fn("generate_greedy")



if __name__ == '__main__':
    # Run the Flask app
    parser = argparse.ArgumentParser(description='Flask server for serving VLLM inference (one TP group)')

    parser.add_argument(
        '--port', 
        type=int, 
        required=True, 
        help='Port number to use (must be an integer)'
    )
    args = parser.parse_args()
    port = args.port

    app.run(host='0.0.0.0', port=port)