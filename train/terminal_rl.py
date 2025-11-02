import asyncio
import os
import sys
import docker
import os
import time
import tempfile
import re
from copy import deepcopy

import torch.distributed as dist
from datasets import load_from_disk
from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.api.workflow_api import RolloutWorkflow
from areal.api.reward_api import AsyncRewardWrapper
from areal.experimental.openai import ArealOpenAI
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.platforms import current_platform
from areal.utils import seeding, stats_tracker
from areal.utils.data import (
    cycle_dataloader,
)
from areal.utils.dataloader import create_dataloader
from areal.utils.device import log_gpu_stats
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from areal.workflow.rlvr import RLVRWorkflow

image_list = {}

def create_image_and_run(index, docker_cmd, model_cmd, python_script_path):
    global image_list
    docker_client = docker.from_env()
    image_name = f"image_{str(index)}"
    has_image = False
    try:
        image = docker_client.images.get(image_name)
        has_image = True
    except docker.errors.ImageNotFound:
        print(f"image for {image_name} not found")
        has_image=False
    if not has_image:
        temp_dir = tempfile.mkdtemp()
        dockerfile_path = os.path.join(temp_dir, "Dockerfile")
        with open(dockerfile_path, 'w') as f:
            f.write(docker_cmd)
            
        image_name = f"image_{str(index)}"
        image, build_logs = docker_client.images.build(path=temp_dir, tag=image_name, rm=True)
        print(f"build image for docker file {docker_cmd} log: {build_logs}")
        image_list.append(image_name)
    print(f"Run model_cmd {model_cmd}")
    try:
        container = docker_client.containers.run(
                image_name,
                command= model_cmd,
                detach=False,
                stdout=True,
                stderr=True,
                remove=True
            )
    except docker.errors.ContainerError as e:
        print(f"ERROR: {str(e)}")
    logs = container.decode('utf-8')
    print(f"{model_cmd}logs are {logs}")
    
    # Run python file to verify result.
    # TODO: Some containers may not have python and pytest installed.
    python_case_pass = False
    python_cmd = "pytest -s " + '/home/zpc/ant/terminalRL/train/' + python_script_path
    try:
        python_output = docker_client.containers.run(
                    image_name,
                    command= python_cmd,
                    detach=False,
                    stdout=True,
                    stderr=True,
                    remove=True,
                    volumes={
                        '/home/zpc/ant/terminalRL/train': {
                            'bind': '/home/zpc/ant/terminalRL/train',
                            'mode': 'rw'
                        }
                    }
                )
        python_case_pass = True
    except Exception as e:
        print(f"ERROR: execute python script failed: {str(e)}")
    return logs, python_case_pass

def terminal_reward_fn(index, result, docker_cmd, test_weights, python_script):
    # Extract command lines from the completion.
    print(f"====result for {index} is {result}")
    pattern = r'<cmd>(.*?)</cmd>'
    matches = re.findall(pattern, result, re.DOTALL)
    if matches:
        # Create Python script to verify the result.
        with open("test_"+index+".py", "w") as f:
            f.write(python_script)
        result, test_case_result = create_image_and_run(index, docker_cmd, matches[0], "test_"+index+".py")
    else:
        result = "FAILED"
        test_case_result = False

    # TODO: Consider test_weights to construct smooth reward output.
    if test_case_result:
        return 1, result
    else:
        return 0, result

def extract_prompt_fn(data):
    return data["prompt"]

class TerminalAgent:
    def __init__(
        self,
        tokenizer,
        max_turns,
    ):
        self.tokenizer = tokenizer
        self.max_turns = max_turns
        # self.async_reward_fn = AsyncRewardWrapper(terminal_reward_fn)

    async def run_agent(self, data, client):
        sys_prompt = "You're a terminal command assistant and you must response with correct command within '<cmd>' and '</cmd>' tag"
        messages = []
        messages.append({"role": "user", "content": sys_prompt + ":" + data["prompt"][:1024]})
        num_turns_left = self.max_turns
        completions = []
        index = data["task_id"]
        while num_turns_left > 0:
            response = await client.chat.completions.create(
                messages=messages,
                temperature=1.0,
                max_completion_tokens=4096,
            )
            completions.append(response)
            content = response.choices[0].message.content
            messages.append({"role": "assistant", "content": content})
            # Get the reward composition and stdout from docker container.
            reward, stdout = terminal_reward_fn(index, content, data["dockerfile"], data["test_weights"], data["test_functions"])
            client.set_reward(response.id, reward)
            if reward == 0:
                print(f"This turn's answer is invalid {index}, stop this trajectory collection.")
                break
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": "Your last command output is " + stdout,
                    }
                )
            num_turns_left -= 1
        return reward


class TerminalMultiturnRLVRWorkflow(RolloutWorkflow):
    def __init__(
        self,
        tokenizer,
        n_trajs,
        max_turns,
    ):
        self.tokenizer = tokenizer
        self.max_turns = max_turns
        self.n_trajs = n_trajs
        self.agent = TerminalAgent(
            tokenizer=tokenizer,
            max_turns=max_turns
        )

    async def arun_episode(self, engine, data):
        clients = [
            ArealOpenAI(engine=engine, tokenizer=self.tokenizer)
            for _ in range(self.n_trajs)
        ]

        # Collect trajectories
        rewards = await asyncio.gather(
            *[
                self.agent.run_agent(
                    data=data,
                    client=clients[i],
                )
                for i in range(self.n_trajs)
            ]
        )
        for reward in rewards:
            stats_tracker.get("rollout").scalar(reward=reward)

        completions_with_reward = {}
        for client in clients:
            client.apply_reward_discount(turn_discount=0.9)
            completions = client.export_completions(style="individual")
            completions_with_reward.update(completions)
            # print(f"client completions is{completions}")
        return completions_with_reward


def main(args):
    config, _ = load_expr_config(args, GRPOConfig)
    config: GRPOConfig

    rank = int(os.getenv("RANK"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train
    assert parallel_strategy is not None

    # Initialize train engine
    actor = FSDPPPOActor(config=config.actor)
    actor.create_process_group(parallel_strategy=parallel_strategy)

    # Create dataset and dataloaders
    train_dataset = load_from_disk(dataset_path=config.train_dataset.path + '/train')
    valid_dataset = load_from_disk(dataset_path=config.valid_dataset.path + '/test')

    train_dataloader = create_dataloader(
        train_dataset,
        rank=actor.data_parallel_rank,
        world_size=actor.data_parallel_world_size,
        dataset_config=config.train_dataset,
    )
    valid_dataloader = create_dataloader(
        valid_dataset,
        rank=actor.data_parallel_rank,
        world_size=actor.data_parallel_world_size,
        dataset_config=config.valid_dataset,
    )
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )

    # Initialize inference engine
    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(train_data_parallel_size=parallel_strategy.dp_size)
    eval_rollout = RemoteSGLangEngine(deepcopy(config.rollout))
    # NOTE: eval does not have any offpolicyness control
    eval_rollout.config.max_head_offpolicyness = int(1e12)
    eval_rollout.initialize()

    weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(allocation_mode)

    actor.initialize(None, ft_spec)
    actor.connect_engine(rollout, weight_update_meta)

    ref = None
    if config.actor.kl_ctl > 0 and config.ref is not None:
        ref = FSDPPPOActor(config=config.ref)
        ref.create_process_group(parallel_strategy=parallel_strategy)
        ref.initialize(None, ft_spec)

    # Create rollout workflow
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)

    workflow = TerminalMultiturnRLVRWorkflow(
        tokenizer=tokenizer,
        n_trajs=2,
        max_turns=2,
    )

    # Run training.
    saver = Saver(config.saver, ft_spec)
    stats_logger = StatsLogger(config, ft_spec)
    evaluator = Evaluator(config.evaluator, ft_spec)

    recover_handler = RecoverHandler(config.recover, ft_spec)
    recover_info = recover_handler.load(
        actor,
        saver,
        evaluator,
        stats_logger,
        train_dataloader,
        inference_engine=rollout,
        weight_update_meta=weight_update_meta,
    )
    start_step = (
        recover_info.last_step_info.next().global_step
        if recover_info is not None
        else 0
    )

    total_epochs = config.total_train_epochs
    steps_per_epoch = len(train_dataloader)
    max_steps = total_epochs * steps_per_epoch

    data_generator = cycle_dataloader(train_dataloader)
    for global_step in range(start_step, max_steps):
        epoch = global_step // steps_per_epoch
        step = global_step % steps_per_epoch
        step_info = StepInfo(
            global_step=global_step,
            epoch=epoch,
            epoch_step=step,
            steps_per_epoch=steps_per_epoch,
        )

        with stats_tracker.record_timing("rollout"):
            if config.async_training:
                batch = actor.prepare_batch(
                    train_dataloader,
                    granularity=actor.config.group_size,
                    workflow=workflow,
                    should_accept=lambda sample: True,
                )
            else:
                batch = actor.rollout_batch(
                    next(data_generator),
                    granularity=actor.config.group_size,
                    workflow=workflow,
                    should_accept=lambda sample: True,
                )

        if config.actor.recompute_logprob or config.actor.use_decoupled_loss:
            with stats_tracker.record_timing("recompute_logp"):
                logp = actor.compute_logp(batch)
                batch["prox_logp"] = logp
                log_gpu_stats("recompute logp")

        if ref is not None:
            with stats_tracker.record_timing("ref_logp"):
                batch["ref_logp"] = ref.compute_logp(batch)
                log_gpu_stats("ref logp")

        with stats_tracker.record_timing("compute_advantage"):
            actor.compute_advantages(batch)
            log_gpu_stats("compute advantages")

        with (
            stats_tracker.record_timing("train_step"),
            stats_tracker.scope("grpo_actor"),
        ):
            stats = actor.ppo_update(batch)
            actor.step_lr_scheduler()
            log_gpu_stats("ppo update")

        # pause inference for updating weights, save, and evaluation
        rollout.pause()

        with stats_tracker.record_timing("update_weights"):
            actor.update_weights(weight_update_meta)

            actor.set_version(global_step + 1)
            rollout.set_version(global_step + 1)
            eval_rollout.set_version(global_step + 1)

        with stats_tracker.record_timing("save"):
            saver.save(actor, epoch, step, global_step, tokenizer=tokenizer)

        with stats_tracker.record_timing("checkpoint_for_recover"):
            recover_handler.dump(
                actor,
                step_info,
                saver,
                evaluator,
                stats_logger,
                train_dataloader,
                tokenizer=tokenizer,
            )

        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        # Upload statistics to the logger (e.g., wandb)
        stats[0].update(
            stats_tracker.export_all(reduce_group=actor.data_parallel_group)
        )
        stats_logger.commit(epoch, step, global_step, stats)

        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        # Resume rollout
        rollout.resume()

    stats_logger.close()
    eval_rollout.destroy()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
