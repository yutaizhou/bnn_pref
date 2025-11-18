"""
Shoutout ReBrac: https://github.com/DT6A/ReBRAC/blob/7f4df8ad83afb0857e276691fc9efb2caa48159c/src/algorithms/rebrac_torch_vis.py
"""

import abc
import os
from collections import deque
from typing import Any, NamedTuple

import ipdb
from tqdm import tqdm

os.environ["MUJOCO_GL"] = "egl"


import dm_env
import h5py
import numpy as np
from dm_control import manipulation, suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import StepType, specs

step_type_lookup = {0: StepType.FIRST, 1: StepType.MID, 2: StepType.LAST}


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key="pixels"):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(
            shape=np.concatenate(
                [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0
            ),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name="observation",
        )

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(
            wrapped_action_spec.shape,
            dtype,
            wrapped_action_spec.minimum,
            wrapped_action_spec.maximum,
            "action",
        )

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(
            observation=time_step.observation,
            step_type=time_step.step_type,
            action=action,
            reward=time_step.reward or 0.0,
            discount=time_step.discount or 1.0,
        )

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


def make(
    name,
    frame_stack,
    action_repeat,
    seed,
    distracting_mode: str = None,
    multitask_mode: str = None,
):
    pixel_hw = 84
    if "offline" in name:
        name = "_".join(name.split("_")[1:3])
    domain, task = name.split("_", 1)
    # overwrite cup to ball_in_cup
    domain = dict(cup="ball_in_cup").get(domain, domain)

    # make sure reward is not visualized
    if multitask_mode is None:
        if (domain, task) in suite.ALL_TASKS:
            env = suite.load(
                domain, task, task_kwargs={"random": seed}, visualize_reward=False
            )
            pixels_key = "pixels"
        else:
            name = f"{domain}_{task}_vision"
            env = manipulation.load(name, seed=seed)
            pixels_key = "front_close"

    # add wrappers
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    # add renderings for clasical tasks
    if (domain, task) in suite.ALL_TASKS:
        # zoom in camera for quadruped
        camera_id = dict(quadruped=2).get(domain, 0)
        render_kwargs = dict(height=pixel_hw, width=pixel_hw, camera_id=camera_id)

        env = pixels.Wrapper(env, pixels_only=True, render_kwargs=render_kwargs)
    # stack several frames
    env = FrameStackWrapper(env, frame_stack, pixels_key)
    env = ExtendedTimeStepWrapper(env)
    return env


class AbstractReplayBuffer(abc.ABC):
    @abc.abstractmethod
    def add(self, time_step):
        pass

    @abc.abstractmethod
    def __next__(
        self,
    ):
        pass

    @abc.abstractmethod
    def __len__(
        self,
    ):
        pass


class EfficientReplayBuffer(AbstractReplayBuffer):
    """Fast + efficient replay buffer implementation in numpy."""

    def __init__(
        self,
        buffer_size,
        batch_size,
        nstep,
        discount,
        frame_stack,
        data_specs=None,
        sarsa=False,
    ):
        self.buffer_size = buffer_size
        self.data_dict = {}
        self.index = -1
        self.traj_index = 0
        self.frame_stack = frame_stack
        self._recorded_frames = frame_stack + 1
        self.batch_size = batch_size
        self.nstep = nstep
        self.discount = discount
        self.full = False
        self.discount_vec = np.power(
            discount, np.arange(nstep)
        )  # n_step - first dim should broadcast
        self.next_dis = discount**nstep
        self.sarsa = sarsa

    def _initial_setup(self, time_step):
        self.index = 0
        self.obs_shape = list(time_step.observation.shape)
        self.ims_channels = self.obs_shape[0] // self.frame_stack
        self.act_shape = time_step.action.shape

        self.obs = np.zeros(
            [self.buffer_size, self.ims_channels, *self.obs_shape[1:]], dtype=np.uint8
        )
        self.act = np.zeros([self.buffer_size, *self.act_shape], dtype=np.float32)
        self.rew = np.zeros([self.buffer_size], dtype=np.float32)
        self.dis = np.zeros([self.buffer_size], dtype=np.float32)
        self.valid = np.zeros([self.buffer_size], dtype=np.bool_)

    def add_data_point(self, time_step):
        first = time_step.first()
        latest_obs = time_step.observation[-self.ims_channels :]
        if first:
            end_index = self.index + self.frame_stack
            end_invalid = end_index + self.frame_stack + 1
            if end_invalid > self.buffer_size:
                if end_index > self.buffer_size:
                    end_index = end_index % self.buffer_size
                    self.obs[self.index : self.buffer_size] = latest_obs
                    self.obs[0:end_index] = latest_obs
                    self.full = True
                else:
                    self.obs[self.index : end_index] = latest_obs
                end_invalid = end_invalid % self.buffer_size
                self.valid[self.index : self.buffer_size] = False
                self.valid[0:end_invalid] = False
            else:
                self.obs[self.index : end_index] = latest_obs
                self.valid[self.index : end_invalid] = False
            self.index = end_index
            self.traj_index = 1
        else:
            np.copyto(self.obs[self.index], latest_obs)  # Check most recent image
            np.copyto(self.act[self.index], time_step.action)
            self.rew[self.index] = time_step.reward
            self.dis[self.index] = time_step.discount
            self.valid[(self.index + self.frame_stack) % self.buffer_size] = False
            if self.traj_index >= self.nstep:
                self.valid[(self.index - self.nstep + 1) % self.buffer_size] = True
            self.index += 1
            self.traj_index += 1
            if self.index == self.buffer_size:
                self.index = 0
                self.full = True

    def add(self, time_step):
        if self.index == -1:
            self._initial_setup(time_step)
        self.add_data_point(time_step)

    def __next__(
        self,
    ):
        indices = np.random.choice(self.valid.nonzero()[0], size=self.batch_size)
        return self.gather_nstep_indices(indices)

    def gather_nstep_indices(self, indices):
        n_samples = indices.shape[0]
        all_gather_ranges = (
            np.stack(
                [
                    np.arange(indices[i] - self.frame_stack, indices[i] + self.nstep)
                    for i in range(n_samples)
                ],
                axis=0,
            )
            % self.buffer_size
        )
        gather_ranges = all_gather_ranges[:, self.frame_stack :]  # bs x nstep
        obs_gather_ranges = all_gather_ranges[:, : self.frame_stack]
        nobs_gather_ranges = all_gather_ranges[:, -self.frame_stack :]

        all_rewards = self.rew[gather_ranges]

        # Could implement below operation as a matmul in pytorch for marginal additional speed improvement
        rew = np.sum(all_rewards * self.discount_vec, axis=1, keepdims=True)

        obs = np.reshape(self.obs[obs_gather_ranges], [n_samples, *self.obs_shape])
        nobs = np.reshape(self.obs[nobs_gather_ranges], [n_samples, *self.obs_shape])

        act = self.act[indices]
        dis = np.expand_dims(
            self.next_dis * self.dis[nobs_gather_ranges[:, -1]], axis=-1
        )

        if self.sarsa:
            nact = self.act[indices + self.nstep]
            return {
                "observations": obs,
                "actions": act,
                "rewards": rew,
                "discounts": dis,
                "next_observations": nobs,
                "next_actions": nact,
            }
            # return (obs, act, rew, dis, nobs, nact)

        return {
            "observations": obs,
            "actions": act,
            "rewards": rew,
            "discounts": dis,
            "next_observations": nobs,
        }
        # return (obs, act, rew, dis, nobs)

    def __len__(self):
        if self.full:
            return self.buffer_size
        else:
            return self.index

    def get_train_and_val_indices(self, validation_percentage):
        all_indices = self.valid.nonzero()[0]
        num_indices = all_indices.shape[0]
        num_val = int(num_indices * validation_percentage)
        np.random.shuffle(all_indices)
        val_indices, train_indices = np.split(all_indices, [num_val])
        return train_indices, val_indices

    def get_obs_act_batch(self, indices):
        n_samples = indices.shape[0]
        obs_gather_ranges = (
            np.stack(
                [
                    np.arange(indices[i] - self.frame_stack, indices[i])
                    for i in range(n_samples)
                ],
                axis=0,
            )
            % self.buffer_size
        )
        obs = np.reshape(self.obs[obs_gather_ranges], [n_samples, *self.obs_shape])
        act = self.act[indices]
        return obs, act


def get_timestep_from_idx(offline_data: dict, idx: int):
    return ExtendedTimeStep(
        step_type=step_type_lookup[offline_data["step_type"][idx]],
        reward=offline_data["reward"][idx],
        observation=offline_data["observation"][idx],
        discount=offline_data["discount"][idx],
        action=offline_data["action"][idx],
    )


def add_offline_data_to_buffer(
    offline_data: dict, replay_buffer: EfficientReplayBuffer, framestack: int = 3
):
    offline_data_length = offline_data["reward"].shape[0]
    for v in offline_data.values():
        assert v.shape[0] == offline_data_length
    for idx in range(offline_data_length):
        time_step = get_timestep_from_idx(offline_data, idx)
        if not time_step.first():
            stacked_frames.append(time_step.observation)
            time_step_stack = time_step._replace(
                observation=np.concatenate(stacked_frames, axis=0)
            )
            replay_buffer.add(time_step_stack)
        else:
            stacked_frames = deque(maxlen=framestack)
            while len(stacked_frames) < framestack:
                stacked_frames.append(time_step.observation)
            time_step_stack = time_step._replace(
                observation=np.concatenate(stacked_frames, axis=0)
            )
            replay_buffer.add(time_step_stack)


def load_offline_dataset_into_buffer(
    offline_dir, replay_buffer, frame_stack, replay_buffer_size
):
    filenames = sorted(offline_dir.glob("*.hdf5"))
    num_steps = 0
    for filename in tqdm(
        filenames,
        desc="Loading offline dataset into replay buffer",
        unit="file",
    ):
        try:
            episodes = h5py.File(filename, "r")
            episodes = {k: episodes[k][:] for k in episodes.keys()}
            add_offline_data_to_buffer(episodes, replay_buffer, framestack=frame_stack)
            ipdb.set_trace()
            length = episodes["reward"].shape[0]
            num_steps += length
        except Exception as e:
            print(f"Could not load episode {str(filename)}: {e}")
            continue
        if num_steps >= replay_buffer_size:
            break
        # print("early break!")
        # break
    print(f"Finished, loaded {num_steps} offline timesteps.")
