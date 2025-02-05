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

"""
Initial goal:

1. load the model for inference from nemo model

Given a nemo model, convert to HF state dict

2. destroy the model after generate
3. rebuild the llm in refit

"""


import secrets
import torch
from abc import ABC
from nemo_aligner.experimental.grpo.inference.base import InferenceBackendBase
from tensorrt_llm import SamplingParams
from tensorrt_llm._torch import LLM
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm.llmapi.llm_utils import CalibConfig
from tensorrt_llm.llmapi.utils import get_total_gpu_memory
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig
from tensorrt_llm.llmapi import KvCacheConfig
from nemo_aligner.experimental.grpo.utils.parallel_state import inference_reshard_region, get_data_parallel_rank, get_data_parallel_world_size, get_model_parallel_src_rank, get_model_parallel_group, get_data_parallel_world_size, get_pipeline_model_parallel_world_size, get_pipeline_model_parallel_group

class GPTGenerateTRTLLMPytorch(InferenceBackendBase, ABC):
    """
    Implements InferenceBackendBase using PyTorch-based high-level APIs for GPT generation using TensorRT-LLM.
    """

    DEFAULT_PAD_ID = -42  # Negative pad ID reserved for cases without pad tokens.

    def __init__(
        self,
        model_cfg,
        end_strings,
        tokenizer,
        sample_temperature=1.0,
        sample_top_k=0,
        sample_top_p=1.0,
        repetition_penalty=1.0,
        max_generation_length=1024,
        max_input_len=1024,
        generation_batch_size=4,
        use_greedy=False,
        trt_model_type="llama",
        seed=None,
        unload_engine_train=False,
        reshard_model=False,
        trt_model_dir="/tmp/trt_llm_model",
    ):

        # Sanity checks for input arguments.
        assert max_input_len > 0, "max_input_len should be greater than 0."
        assert max_generation_length > 0, "max_generation_length should be greater than 0."
        assert (
            max_input_len + max_generation_length <= model_cfg.encoder_seq_length
        ), f"We require max_input_len ({max_input_len}) + max_generation_length ({max_generation_length}) <= model_cfg.encoder_seq_length ({model_cfg.encoder_seq_length})"

        self.model_cfg = model_cfg
        self.tokenizer = tokenizer
        self.max_generation_length = max_generation_length
        self.max_input_len = max_input_len
        self.generation_batch_size = generation_batch_size
        self.trt_model_dir = trt_model_dir
        self.trt_model_type = trt_model_type
        self.unload_engine_train = unload_engine_train

        # Set up random seed for generating random numbers.
        rng_generator = torch.Generator(device="cpu")
        seed = secrets.randbits(32) if seed is None else seed
        rng_generator.manual_seed(seed)
        self.rng_generator = rng_generator

        # Define the pad_id and eos_id.
        self.pad_id = GPTGenerateTRTLLMPytorch.DEFAULT_PAD_ID
        self.eos_id = tokenizer.eos_id

        # Set up sampling parameters.
        self.sampling_params = SamplingParams(
            end_id=self.eos_id,
            pad_id=self.pad_id,
            temperature=sample_temperature,
            top_k=sample_top_k,
            top_p=sample_top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_generation_length,
            seed=seed,
            use_beam_search=use_greedy,
        )

    def refit(self, model):
        """
        Refit or recompile the model. For the high-level API, rebuild the LLM.
        """
        
        # Get parallel state information
        from megatron.core import parallel_state
        dp_rank = parallel_state.get_data_parallel_rank()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        tp_group = parallel_state.get_tensor_model_parallel_group()
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        pp_first_rank = parallel_state.get_pipeline_model_parallel_first_rank()
        pp_last_rank = parallel_state.get_pipeline_model_parallel_last_rank()
        pp_size = parallel_state.get_pipeline_model_parallel_world_size()
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        pp_is_last = parallel_state.is_pipeline_last_stage(ignore_virtual=True)
        pp_is_first = parallel_state.is_pipeline_first_stage(ignore_virtual=True)
        vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

        print(f"dp_rank {dp_rank}, tp_rank {tp_rank}, tp_size {tp_size}, tp_group {tp_group}, pp_rank {pp_rank}, pp_first_rank {pp_first_rank}, pp_last_rank {pp_last_rank} pp_size {pp_size}, pp_group {pp_group}, pp_is_last {pp_is_last}, pp_is_first {pp_is_first}, vp_size {vp_size}")       
        # Since we're not using pipeline parallelism
        inference_pp_size = 1
        
        # Get model config and vocab size
        nemo_config = self.model_cfg
        tokenizer_vocab_size = self.tokenizer.vocab_size
        # from nemo.export.trt_llm.converter.model_to_trt_llm_ckpt import dist_model_to_trt_llm_ckpt
        
        # try:
        #     # Convert nemo model weights to TRT-LLM format
        #     weights_dict = dist_model_to_trt_llm_ckpt(
        #         model=model,
        #         nemo_model_config=nemo_config,
        #         inference_tp_size=tp_size,
        #         inference_pp_size=inference_pp_size,
        #         tokenizer_vocab_size=tokenizer_vocab_size,
        #         fp8_quantized=False,  # Add as parameter if needed
        #         fp8_kvcache=False,    # Add as parameter if needed
        #     )
        #     print(type(weights_dict))
        #     print(weights_dict.keys())
        # except Exception as e:
        #     print(f"Error in dist_model_to_trt_llm_ckpt: {e}")
            
        print(f"Process {torch.distributed.get_rank()} completed weight conversion")        
        # Make sure weight conversion is complete on all processes
        torch.distributed.barrier()

        # Split the MPI communicator
        from mpi4py import MPI

        # Get the global rank from MPI
        global_rank = MPI.COMM_WORLD.Get_rank()

        # Split communicator into groups of size tp_size
        new_color = global_rank // tp_size      # e.g., 0 for ranks 0-3 and 1 for ranks 4-7
        new_key   = global_rank % tp_size         # e.g., 0,1,2,3 for local ordering within the group

        # Create a new communicator with remapped ranks
        local_comm = MPI.COMM_WORLD.Split(color=new_color, key=new_key)

        # Optionally, if LLM initialization requires the global communicator to be remapped:
        MPI.COMM_WORLD = local_comm
 
        print(f"Global rank {global_rank} becomes local rank {MPI.COMM_WORLD.Get_rank()} in group {new_color}")        
        # Make sure all processes have completed the MPI split
        torch.distributed.barrier()
        
        print(f"Process {torch.distributed.get_rank()} starting weight conversion")

        # Only tp_rank 0 initializes the LLM
        if tp_rank == 0:
            print(f"Process {torch.distributed.get_rank()} starting LLM initialization")
            self.pytorch_config = PyTorchConfig(
                use_cuda_graph=False,
                cuda_graph_max_batch_size=self.generation_batch_size,
            )
            
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', trust_remote_code=True)
            print(f"Process {torch.distributed.get_rank()} tokenizer loaded")
            print(f"Process {torch.distributed.get_rank()} starting LLM initialization")
            self.llm = LLM(
                # model='meta-llama/Meta-Llama-3-8B-Instruct',
                model='/lustre/fsw/portfolios/coreai/users/pchadha/inference-backend-experiments/hub/models--meta-llama--Llama-3.1-8B-Instruct',
                tokenizer=tokenizer,
                tensor_parallel_size=tp_size,
                pytorch_backend_config=self.pytorch_config,
            )
            print(f"Process {torch.distributed.get_rank()} LLM initialized")
        # Final sync
        print(f"Process {torch.distributed.get_rank()} waiting for final barrier")
        torch.distributed.barrier()
        print(f"Process {torch.distributed.get_rank()} completed refit")
    
    def generate(self, inputs):
        """
        Generate outputs given input tensors.

        Args:
            inputs: Tuple of two tensors (input_tokens, input_lengths) for prompts.

        Returns:
            A dict:
                - 'response_tokens': Generated response tokens.
                - 'response_lengths': Corresponding lengths of the responses.
        """
        prompt_tokens, prompt_lengths = inputs

        # Validate input dimensions.
        assert prompt_tokens.size(1) <= self.max_input_len, "Input tokens exceed max_input_len."

        # # Only DP leader performs generation
        # if self.dp_rank == 0:

        #     # Convert input tensors into PromptInputs format.
        #     prompts = []
        #     for idx in range(prompt_tokens.size(0)):
        #         token_ids = prompt_tokens[idx][: prompt_lengths[idx]].tolist()
        #         prompts.append(PromptInputs(input_ids=token_ids))

        #     # Perform generation using the high-level API.
        #     responses = self.llm.generate(prompts, sampling_params=self.sampling_params, use_tqdm=False)

        #     # Parse and process the outputs.
        #     response_tokens, response_lengths = [], []
        #     for response in responses:
        #         response_tokens.append(response.output_ids)
        #         response_lengths.append(len(response.output_ids))
        #     output = {
        #         "response_tokens": torch.tensor(response_tokens, dtype=torch.int64),
        #         "response_lengths": torch.tensor(response_lengths, dtype=torch.int64),
        #     }

        # else:
        #     # Non-leader DP ranks output empty results
        #     output = {
        #         "response_tokens": torch.empty(0, dtype=torch.int64),
        #         "response_lengths": torch.empty(0, dtype=torch.int64),
        #     }
        
        # # Synchronize all DP ranks after generation
        # torch.distributed.barrier()

        output = {
            "response_tokens": torch.empty(0, dtype=torch.int64),
            "response_lengths": torch.empty(0, dtype=torch.int64),
        }

        return output

    def free(self):
        """
        Free up memory or resources used by the backend.
        """
        pass
        # if self.dp_rank == 0 and self.unload_engine_train:
        #     self.llm.free()
