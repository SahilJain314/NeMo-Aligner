# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
from functools import partial
from collections import deque
from itertools import chain
from typing import Iterable, List, Optional

import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset
from megatron.core.utils import divide
from omegaconf.omegaconf import OmegaConf

from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    BaseMegatronSampler,
)

from nemo_aligner.algorithms.reinforce import ReinforceTrainer
from nemo_aligner.data.nlp.builders import (
    build_dataloader,
    build_train_valid_test_math_datasets,
    math_collate_with_pad_to_max_batch,
)
from nemo_aligner.models.nlp.gpt.megatron_gpt_reinforce_math_actor import MegatronGPTReinforceActorModel
from nemo_aligner.models.nlp.gpt.grader_client import RemoteGraderClient
from nemo_aligner.utils import parallel_state
from nemo_aligner.utils.batch_iterators import get_batch_iterator_cls
from nemo_aligner.utils.distributed import Timer
from nemo_aligner.utils.train_script_utils import (
    CustomLoggerWrapper,
    add_custom_checkpoint_callback,
    extract_optimizer_scheduler_from_ptl_model,
    init_distributed,
    init_peft,
    init_using_ptl,
    resolve_and_create_trainer,
    retrieve_custom_trainer_state_dict,
)
from nemo_aligner.utils.utils import load_and_override_model_config, load_from_nemo, retrieve_model_state_dict_in_cpu
import time

"""Script to start REINFORCE training"""

OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver("int_div", lambda x, y: x // y, replace=True)
OmegaConf.register_new_resolver("subtract", lambda x, y: x - y, replace=True)

mp.set_start_method("spawn", force=True)

class CombinedDataset(Dataset):
    def __init__(self, main_dataset, hard_problems_list):
        self.main_dataset = main_dataset
        assert isinstance(hard_problems_list, list) or isinstance(hard_problems_list, deque)
        self.hard_samples = list(hard_problems_list)
        self.hard_sample_indices_to_remove = []
        self.hard_samples_length_at_last_getitem = len(self.hard_samples)

        
    def __len__(self):
        # Return the combined length, dynamically adjusts as hard_problems_list changes
        return len(self.main_dataset) + len(self.hard_problems_list)

    def add_hard_samples(self, hard_samples: List[int]) -> None:
        """Add newly discovered hard samples to the queue."""
        self.hard_samples += hard_samples

    def remove_used_hard_samples(self) -> None:
        """Remove hard samples that have been sampled from the queue."""
        assert len(self.hard_samples) == self.hard_samples_length_at_last_getitem, \
            "hard_samples length changed since last __getitem__ call. This will make indices incorrect."
        self.hard_samples = [x for i, x in enumerate(self.hard_samples) 
                           if i not in self.hard_sample_indices_to_remove]
        self.hard_sample_indices_to_remove = []

    def __getitem__(self, idx):
        if idx < len(self.main_dataset):
            return self.main_dataset[idx]
        else:
            # Fetch from hard problems, adjust index accordingly
            hard_idx = idx - len(self.main_dataset)
            self.hard_sample_indices_to_remove.append(hard_idx)
            self.hard_samples_length_at_last_getitem = len(self.hard_samples)
            return self.hard_samples[hard_idx]
        
class MegatronPretrainingDynamicSampler(BaseMegatronSampler):
    """
    A dynamic sampler that:
      - Reserves a fixed portion (hard_sample_ratio) of each micro-batch for hard samples.
      - Fills the remainder with normal samples from [consumed_samples..total_samples).
      - Any discovered "hard samples" can be queued via `add_hard_samples`.
      - Works for multiple epochs by simply re-calling __iter__() each epoch (like standard PyTorch samplers).

    The base class is unchanged. We only add logic here.
    """

    def __init__(
        self,
        total_samples: int,
        consumed_samples: int,
        micro_batch_size: int,
        data_parallel_rank: int,
        data_parallel_size: int,
        drop_last: bool = True,
        global_batch_size: Optional[int] = None,
        rampup_batch_size: Optional[list] = None,
        pad_samples_to_global_batch_size: Optional[bool] = False,
        hard_sample_ratio: float = 0.4,  # Default 40% of each micro-batch is reserved for hard samples
    ):
        super().__init__(
            total_samples=total_samples,
            consumed_samples=consumed_samples,
            micro_batch_size=micro_batch_size,
            data_parallel_rank=data_parallel_rank,
            data_parallel_size=data_parallel_size,
            drop_last=drop_last,
            global_batch_size=global_batch_size,
            rampup_batch_size=rampup_batch_size,
            pad_samples_to_global_batch_size=pad_samples_to_global_batch_size,
        )
        self.nb_hard_samples = 0
        # A ratio [0,1] indicating what fraction of each micro-batch we aim to fill with hard samples
        self.hard_sample_ratio = hard_sample_ratio

    def add_nb_hard_samples(self, nb_hard_samples: int) -> None:
        self.nb_hard_samples += nb_hard_samples
        
    def get_start_end_idx(self):
        """
        The slice within each micro-batch that this data-parallel rank should receive.
        """
        start_idx = self.data_parallel_rank * self.micro_batch_size
        end_idx = start_idx + self.micro_batch_size
        return start_idx, end_idx

    def _get_padding_indices(self, pad_samples_num):
        """
        Provide 'fake' indices for padding if needed (e.g. -1, -2, ...).
        """
        return range(-1, -pad_samples_num - 1, -1)

    def __iter__(self) -> Iterable[List[int]]:
        # Normal sample range
        normal_indices = range(self.consumed_samples, self.total_samples)

        # Possibly pad to a global_batch_size if drop_last=False
        if (not self.drop_last) and self.pad_samples_to_global_batch_size:
            num_available = self.total_samples - self.consumed_samples
            pad_samples_num = -num_available % self.global_batch_size
            if pad_samples_num > 0:
                pad_indices = self._get_padding_indices(pad_samples_num)
                normal_indices = chain(normal_indices, pad_indices)

        normal_indices_iter = iter(normal_indices)

        batch = []
        # We know the total size of each micro-batch across all ranks
        total_batch_size = self.micro_batch_times_data_parallel_size
        # How many hard samples we *want* in each micro-batch
        desired_hard_samples = int(total_batch_size * self.hard_sample_ratio)

        while True:
            # 1) Fill the "reserved" portion with hard samples (or as many as the queue has)
            batch_hard = []
            if desired_hard_samples <= self.nb_hard_samples:
                self.nb_hard_samples -= desired_hard_samples
                batch_hard.append([i for i in range(desired_hard_samples)])
            else:
                batch_hard.append([i for i in range(self.nb_hard_samples)])
                self.nb_hard_samples = 0
            
            # 2) Fill the remainder with normal samples until total_batch_size is reached
            batch_regular = []
            while (len(batch_hard) + len(batch_regular)) < total_batch_size:
                try:
                    idx = next(normal_indices_iter)
                    batch_regular.append(idx)
                except StopIteration:
                    # Out of normal samples
                    break

            # Combine
            batch = batch_hard + batch_regular

            # If we formed a full micro-batch, yield it
            if len(batch) == total_batch_size:
                start_idx, end_idx = self.get_start_end_idx()
                yield batch[start_idx:end_idx]
                batch = []
            else:
                # Partial or empty. If partial and drop_last=False, yield it
                if len(batch) > 0 and not self.drop_last:
                    # Shouldn't happen if we pad to global batch size
                    assert not self.pad_samples_to_global_batch_size, (
                        "With pad_samples_to_global_batch_size=True, you shouldn't "
                        "encounter partial micro-batches."
                    )
                    start_idx, end_idx = self.get_start_end_idx()
                    yield batch[start_idx:end_idx]
                # Done for this pass (epoch)
                break

def build_custom_dataloader(
    cfg,
    dataset,
    consumed_samples,
    mbs,
    gbs,
    drop_last=True,
    pad_samples_to_global_batch_size=False,
    collate_fn=None,
    load_gbs=True,
    use_random_sampler=True,
    hard_problems_list=None,
    hard_sample_ratio=0.4 
):
    """Buld dataloader given an input dataset."""
    from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
        MegatronPretrainingBatchSampler,
        MegatronPretrainingRandomBatchSampler,
    )

    from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
        MegatronPretrainingRandomSampler,
        MegatronPretrainingSampler,
    )

    combined_dataset = CombinedDataset(dataset, hard_problems_list)
    logging.info(f"Building dataloader with consumed samples: {consumed_samples}")

    # Common parameters for batch sampler creation
    common_params = {
        "total_samples": len(combined_dataset),
        "consumed_samples": consumed_samples,
        "micro_batch_size": mbs,
        "data_parallel_rank": parallel_state.get_data_parallel_rank(),
        "data_parallel_size": parallel_state.get_data_parallel_world_size(),
        "drop_last": drop_last,
        "global_batch_size": gbs,
        "pad_samples_to_global_batch_size": pad_samples_to_global_batch_size,
    }

    if use_random_sampler:
        cls = MegatronPretrainingRandomBatchSampler if load_gbs else MegatronPretrainingRandomSampler
        common_params["seed"] = cfg.model.seed
    else:
        common_params["hard_sample_ratio"] = hard_sample_ratio
        cls = MegatronPretrainingBatchSampler if load_gbs else MegatronPretrainingDynamicSampler#MegatronPretrainingSampler

    batch_sampler = cls(**common_params)

    return torch.utils.data.DataLoader(
        combined_dataset,
        batch_sampler=batch_sampler,
        num_workers=cfg.model.data.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

@hydra_runner(config_path="conf", config_name="gpt_reinforce_actor")
def main(cfg) -> None:
    print('START')
    cfg.model = load_and_override_model_config(cfg.pretrained_checkpoint.restore_from_path, cfg.model)

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    trainer = resolve_and_create_trainer(cfg, "reinforce")

    print('PTL TRAINER BUILT')
    exp_manager(trainer, cfg.exp_manager)

    logger = CustomLoggerWrapper(trainer.loggers)

    ptl_model = load_from_nemo(
        MegatronGPTReinforceActorModel,
        cfg.model,
        trainer,
        strict=True,
        restore_path=cfg.pretrained_checkpoint.restore_from_path,
    )
    print(time.time(), 'MODEL BUILT')

    init_peft(ptl_model, cfg.model)

    init_policy_state_dict = None

    # only need this if we are running with inital kl penalty & full-parameter tuning
    if cfg.trainer.reinforce.initial_policy_kl_penalty > 0 and cfg.model.peft.peft_scheme == "none":
        init_policy_state_dict = retrieve_model_state_dict_in_cpu(
            ptl_model, megatron_amp_O2=cfg.model.get("megatron_amp_O2", False)
        )

    ptl_model.init_policy_state_dict = init_policy_state_dict

    # pull values from checkpoint
    trainer_restore_path = trainer.ckpt_path

    # TODO: log this restore path
    if trainer_restore_path is not None:
        custom_trainer_state_dict = retrieve_custom_trainer_state_dict(trainer)
    else:
        custom_trainer_state_dict = None

    print(time.time(), 'PRE DISTR INIT')
    init_distributed(trainer, ptl_model, cfg.model.get("transformer_engine", False))
    print('DISTR INITd')

    # use the entire dataset
    train_valid_test_num_samples = [-1, -1, -1]
    train_ds, validation_ds, _ = build_train_valid_test_math_datasets(
        cfg=cfg.model,
        data_prefix=cfg.model.data.data_prefix,
        data_impl=cfg.model.data.data_impl,
        splits_string=cfg.model.data.splits_string,
        train_valid_test_num_samples=train_valid_test_num_samples,
        seq_length=cfg.model.data.seq_length,
        seed=cfg.model.seed,
        tokenizer=ptl_model.tokenizer,
    )

    max_seqlen = cfg.model.reinforce.length_params.max_length
    eos_id = ptl_model.tokenizer.eos_id

    # collate fn to pad to the max seq length in the batch
    collate_fn = math_collate_with_pad_to_max_batch(max_seqlen, eos_id, cfg, generate_masks_and_position_ids=False)

    train_dataloader_builder = partial(
        build_custom_dataloader,
        cfg=cfg,
        dataset=train_ds,
        mbs=cfg.model.reinforce.rollout_micro_batch_size,
        gbs=cfg.model.reinforce.num_rollout_samples,
        collate_fn=collate_fn,
        load_gbs=False,
        use_random_sampler=cfg.model.data.shuffle_train_data,
    )

    val_dataloader_builder = partial(
        build_custom_dataloader,
        cfg=cfg,
        dataset=validation_ds,
        mbs=cfg.model.reinforce.val_rollout_micro_batch_size,
        gbs=cfg.model.reinforce.num_val_samples,
        collate_fn=collate_fn,
        load_gbs=False,
        use_random_sampler=False,
    )
    print(time.time(), 'DATA DONE')

    # nemo uses the train dataloader to figure out
    # max steps to take when max_steps = -1
    # but our train dataloader is for the prompts
    # so we instaniate a dummy dataloader
    # to get the proper max *optimization* steps
    # nemo treats batch size of normal dataloader as GBS/DP
    # so we need to offset it by DP
    dummy_train_dataloader = torch.utils.data.DataLoader(
        dataset=train_ds, batch_size=divide(cfg.model.global_batch_size, parallel_state.get_data_parallel_world_size())
    )

    print(time.time(), 'PRE INIT USING PTL')
    logging.warning("PRE")
    init_using_ptl(trainer, ptl_model, dummy_train_dataloader, train_ds)
    logging.warning("POST")
    print(time.time(), 'POST INIT USING PTL')
    # make sure the dummy train dataloader is never used
    del ptl_model._train_dl
    del dummy_train_dataloader

    optimizer, scheduler = extract_optimizer_scheduler_from_ptl_model(ptl_model)
    ckpt_callback = add_custom_checkpoint_callback(trainer, ptl_model)

    logger.log_hyperparams(OmegaConf.to_container(cfg))

    print('OPTIM EXTRACTED')
    rm = RemoteGraderClient(cfg.remote_rm)
    timer = Timer(cfg.exp_manager.get("max_time_per_run") if cfg.exp_manager else None)

    batch_iterator_cfg = cfg.trainer.reinforce.get("batch_iterator", {})
    batch_iterator_cls = get_batch_iterator_cls(batch_iterator_cfg)

    reinforce_trainer = ReinforceTrainer(
        cfg=cfg.trainer.reinforce,
        model=ptl_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader_builder=train_dataloader_builder,
        val_dataloader_builder=val_dataloader_builder,
        collate_fn=collate_fn,
        rm=rm,
        batch_iterator_cls=batch_iterator_cls,
        logger=logger,
        ckpt_callback=ckpt_callback,
        run_timer=timer,
    )
    print(time.time(), 'TRAINER BUILT')

    if custom_trainer_state_dict is not None:
        reinforce_trainer.load_state_dict(custom_trainer_state_dict)
    print('LOAD STATE DICT DONE')

    print(time.time(), 'CALLING FIT')
    reinforce_trainer.fit()

    # Note: The main loop creates multiple HTTPCommunicators which own a
    # pytriton.client.FuturesModelClient. At the end of the loop, we manually
    # close all FuturesModelClients since we do not use the context manager
    # syntax. This guarantees all dangling threads are no longer blocking.
    # `atexit` does not suffice since the registered cleanup function can be
    # queued behind another blocking atexit registered function.
    # TODO: utilize context managers to avoid manual cleanup
    rm.communicator.close()


if __name__ == "__main__":
    main()
