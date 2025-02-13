import argparse
import os
from flask import Flask, request, jsonify
import multiprocessing as mp
import torch

app = Flask(__name__)

# ctx = mp.get_context("spawn")
# in_q = ctx.Queue()
# out_q = ctx.Queue()
# inference_process = None

# def worker_process(in_queue, out_queue, load_path, tp):
#     import torch
#     import gc

#     from tensorrt_llm._torch import LLM
#     from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
#     # torch.multiprocessing.set_start_method("spawn", force=True)

#     class TRTLLMPytorchInferenceServer:
#         def __init__(self) -> None:
#             self.running = False
#             self.llm = None
        
#         def start(self, path, tp):
#             print(f"starting llm server", flush=True)
#             pytorch_config = PyTorchConfig(
#                 use_cuda_graph=False,
#                 #cuda_graph_max_batch_size=bs,
#             )

#             self.llm = LLM(model=path, tensor_parallel_size=tp, pytorch_backend_config=pytorch_config)
#             self.running = True
#             print(f"TRTLLM Pytorch inference server started.")

#         def shutdown(self):
#             self.llm.shutdown()
#             del self.llm
#             gc.collect()
#             torch.cuda.empty_cache()
#             self.running = False
#             print("TRTLLM Pytorch Server shutdown.")

#         def generate(self, batch_tokens):
#             """
#             For each sequence in batch_tokens, we generate:
#               - a new sequence of output tokens
#               - a list of logprobs (one per token in the generated sequence)
#             """
#             from tensorrt_llm import SamplingParams
#             from tensorrt_llm.inputs.data import TokensPrompt

#             sampling_params = SamplingParams(
#                 temperature=1.0,
#                 top_p=1.0,
#                 max_tokens=2048,
#                 return_log_probs=True,
#             )

#             # Generate texts from the prompts. The output is a list of RequestOutput objects
#             # that contain the prompt, generated text, and other information.
#             prompt_tokens = [TokensPrompt(prompt_token_ids=tok_seq) for tok_seq in batch_tokens]
#             outputs = self.llm.generate(prompt_tokens, sampling_params, use_tqdm=True)
#             logprobs = []
#             out_tokens = []
#             for output in outputs:
#                 # lps = [next(iter(l.items()))[1].logprobs for l in output.outputs[0].logprobs]
#                 # out_toks = [next(iter(l.items()))[0] for l in output.outputs[0].logprobs]
#                 lps = output.outputs[0].logprobs
#                 out_toks = output.outputs[0].token_ids
#                 logprobs.append(lps)
#                 out_tokens.append(out_toks)

#             return out_tokens, logprobs
        

#     server = TRTLLMPytorchInferenceServer()
#     server.start(load_path, tp)

#     while True:
#         command, args = in_queue.get()
#         if command == "shutdown":
#             server.shutdown()
#             return
#         elif command == "generate":
#             batch_tokens = args[0]
#             out_queue.put(server.generate(batch_tokens))
#         else:
#             raise NotImplementedError("Unknown Command")



# @app.route('/start', methods=['POST'])
# def start_server():
#     """
#     Starts the vLLM inference server. Returns only when the server is ready.
#     """
#     global inference_process, in_q, out_q, ctx
#     print(f"start call")

#     if inference_process is not None:
#         in_q.put(("shutdown", None))
#         inference_process.join()
#         inference_process = None

#         ctx = mp.get_context("spawn")
#         in_q = ctx.Queue()
#         out_q = ctx.Queue()

#     data = request.get_json()
#     inference_process = mp.Process(target=worker_process, args=(in_q, out_q, data["checkpoint_path"], data["tp"]))
#     inference_process.start()
#     return jsonify({"status": "started"}), 200

#     # if inference_process is not None:
#     #     return jsonify({"status": "already running"}), 200

#     # data = request.get_json()
#     # # set CUDA_VISIBLE_DEVICES to the TP group (str)
#     # # os.environ["CUDA_VISIBLE_DEVICES"]=",".join(map(str, list(range(data["tp_src_gpu_idx"], data["tp_src_gpu_idx"]+data["tp"]))))
#     # # print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
#     # inference_process = mp.Process(target=worker_process, args=(in_q, out_q, data["checkpoint_path"], data["tp"]))
#     # inference_process.start()
#     # return jsonify({"status": "started"}), 200


# @app.route('/shutdown', methods=['POST'])
# def shutdown_server():
#     """
#     Shuts down the vLLM inference server.
#     """
#     #global inference_server
#     global inference_process
#     global in_q
#     print(f"shutdown call")

#     # if inference_server is None or not inference_server.running:
#     #    return jsonify({"status": "inference server not running"}), 400

#     if inference_process is not None:
#         in_q.put(("shutdown", None))
#         inference_process.join()
#         inference_process = None
#         return jsonify({"status": "shutdown successful"}), 200
#     #inference_server.shutdown()
#     return jsonify({"status": "server not running"}), 400


# @app.route('/generate', methods=['POST'])
# def generate():
#     """
#     Performs inference on a batch of token lists.
#     Expects a JSON payload that is a list of lists of tokens.
#     Returns:
#       - generations: list of generated token sequences for each batch element
#       - logprobs: list of token-wise log-probabilities for each generation
#     """
#     global in_q
#     global out_q
#     global inference_process
#     print(f"Generate call")
#     if inference_process is None:
#         return jsonify({"error": "inference server not running"}), 400

#     data = request.get_json()
#     if not isinstance(data, list):
#         return jsonify({"error": "Expected a list of lists of tokens"}), 400

#     # Validate that each element in the list is itself a list
#     for i, item in enumerate(data):
#         if not isinstance(item, list):
#             return jsonify({"error": f"Element at index {i} is not a list"}), 400

#     in_q.put(("generate", (data,)))
#     generations, logprobs = out_q.get()
#     response = {
#         "response_tokens": generations,
#         "response_logprobs": logprobs
#     }
#     return jsonify(response), 200



# Global variable to hold the inference server instance
inference_server = None
import torch
import gc
import subprocess

import threading
from tensorrt_llm._torch import LLM
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm.llmapi import KvCacheConfig


# Global lock to guard all API accesses
api_lock = threading.Lock()

class TRTLLMPytorchInferenceServer:
    def __init__(self) -> None:
        self.running = False
        self.llm = None
        self.max_seq_len = None
        self.end_id = None
        self.pad_id = None

    def get_gpu_memory_usage(self):
        command = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
        memory_used = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1]
        for idx, mem in enumerate(memory_used):
            print(f"GPU {idx} using {mem} GB", flush=True)


    def start(self, path, tp):
        self.get_gpu_memory_usage()

        if self.llm is None:
            print(f"starting llm server")
            pytorch_config = PyTorchConfig(
                use_cuda_graph=False,
                # attn_backend = 'VANILLA',
            )
            # self.llm = LLM(model=path, tensor_parallel_size=tp, pytorch_backend_config=pytorch_config, kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0., enable_block_reuse=False))
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(path)
            self.max_seq_len = tokenizer.model_max_length
            self.end_id = tokenizer.eos_token_id
            self.pad_id = tokenizer.pad_token_id
            print(f"Max seq len of model is {self.max_seq_len}")
            print(f"End id of model is {self.end_id}")
            print(f"Pad id of model is {self.pad_id}")

            self.llm = LLM(model=path, tensor_parallel_size=tp, pytorch_backend_config=pytorch_config, kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.7, enable_block_reuse=True))
            self.running = True
        else:
            self.llm.load_model(path)
        print(f"TRTLLM Pytorch inference server started.", flush=True)
        self.get_gpu_memory_usage()

    def shutdown(self):
        self.get_gpu_memory_usage()

        # self.llm.shutdown()
        # del self.llm
        # gc.collect()
        # torch.cuda.empty_cache()
        # self.running = False
        self.llm.free_gpu_resources()
        print("TRTLLM Pytorch gpu resources freed.", flush=True)
        # print the current memory usage for all GPUs
        self.get_gpu_memory_usage()

    def generate(self, batch_tokens):
        from tensorrt_llm import SamplingParams
        from tensorrt_llm.inputs.data import TokensPrompt

        self.get_gpu_memory_usage()


        assert self.max_seq_len is not None, "Max seq len is not set"
        assert self.end_id is not None, "End id is not set"
        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=4096, #self.max_seq_len,
            detokenize=False,
            return_log_probs=True,
            end_id=self.end_id,
            pad_id=self.pad_id,
        )

        prompt_tokens = [TokensPrompt(prompt_token_ids=tok_seq) for tok_seq in batch_tokens]
        print(len(prompt_tokens))
        outputs = self.llm.generate(prompt_tokens, sampling_params, use_tqdm=True)
        logprobs = []
        out_tokens = []
        for output in outputs:
            lps = output.outputs[0].logprobs
            out_toks = output.outputs[0].token_ids
            logprobs.append(lps)
            out_tokens.append(out_toks)

        self.get_gpu_memory_usage()

        return out_tokens, logprobs

@app.route('/start', methods=['POST'])
def start_server():
    global inference_server
    with api_lock:
        print(f"start call", flush=True)
        data = request.get_json()

        if inference_server is None:
            print(f"First time start code path", flush=True)
            inference_server = TRTLLMPytorchInferenceServer()
            inference_server.start(data["checkpoint_path"], data["tp"])
            return jsonify({"status": "started"}), 200
        else:
            print(f"Refit code path", flush=True)
            inference_server.start(data["checkpoint_path"], data["tp"])
            return jsonify({"status": "started"}), 200

@app.route('/shutdown', methods=['POST'])
def shutdown_server():
    global inference_server
    with api_lock:
        print(f"shutdown call", flush=True)
        inference_server.shutdown()
        return jsonify({"status": "shutdown complete"}), 200

@app.route('/generate', methods=['POST'])
def generate():
    global inference_server
    with api_lock:
        print(f"Generate call", flush=True)
        
        if inference_server is None or not inference_server.running:
            return jsonify({"error": "inference server not running"}), 400


        import time
        start_time = time.time()
        data = request.get_json()
        end_time = time.time()
        print(f"Request get_json time: {end_time - start_time} seconds", flush=True)
        if not isinstance(data, list):
            return jsonify({"error": "Expected a list of lists of tokens"}), 400

        for i, item in enumerate(data):
            if not isinstance(item, list):
                return jsonify({"error": f"Element at index {i} is not a list"}), 400

        generations, logprobs = inference_server.generate(data)
        response = {
            "response_tokens": generations,
            "response_logprobs": logprobs
        }

        import time
        start_time = time.time()
        response = jsonify(response), 200
        end_time = time.time()
        print(f"Response jsonify time: {end_time - start_time} seconds", flush=True)
        return response

if __name__ == '__main__':
    # Run the Flask app
    mp.set_start_method("spawn", force=True)
  
    parser = argparse.ArgumentParser(description='Flask server for serving TRTLLM Pytorch inference (one TP group)')

    parser.add_argument(
        '--port', 
        type=int, 
        required=True, 
        help='Port number to use (must be an integer)'
    )
    args = parser.parse_args()
    port = args.port
    num_devices = torch.cuda.device_count()
    print(f"Number of CUDA devices: {num_devices}")

    def print_device_properties():
        # Check CUDA availability
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        # Get device count
        device_count = torch.cuda.device_count()
        print(f"Number of CUDA devices: {device_count}")
        
        # Iterate over all available devices
        for i in range(device_count):
            print(f"\nDevice {i}: {torch.cuda.get_device_properties(i)}")
    # Run the function
    print_device_properties()

    app.run(host='0.0.0.0', port=port)
