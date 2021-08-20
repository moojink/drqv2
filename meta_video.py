# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import cv2
import imageio
import numpy as np
from pathlib import Path


class VideoRecorder:
    def __init__(self, view, root_dir, mode, render_size=256, fps=30):
        self.view = view
        if root_dir is not None:
            if mode == 'eval':
                self.save_dir = root_dir / 'eval_video'
            elif mode == 'test':
                self.save_dir = root_dir / 'test_video'
            else:
                raise ValueError
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        if self.view == 'both':
            self.frames1 = []
            self.frames3 = []
        else:
            self.frames = []

    def init(self, env, enabled=True):
        if self.view == 'both':
            self.frames1 = []
            self.frames3 = []
        else:
            self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(env)

    def record(self, env):
        if self.enabled:
            if self.view == 'both':
                frame1 = env.render_overwrite(offscreen=True, overwrite_view="view_1", resolution=(self.render_size, self.render_size))
                frame1 = np.transpose(frame1, (1, 2, 0)) # e.g. (3, 84, 84) -> (84, 84, 3) bc the latter is needed to save gif
                self.frames1.append(frame1)
                frame3 = env.render_overwrite(offscreen=True, overwrite_view="view_3", resolution=(self.render_size, self.render_size))
                frame3 = np.transpose(frame3, (1, 2, 0)) # e.g. (3, 84, 84) -> (84, 84, 3) bc the latter is needed to save gif
                self.frames3.append(frame3)
            else:
                frame = env.render(offscreen=True, camera_name="configured_view", resolution=(self.render_size, self.render_size))
                frame = np.transpose(frame, (1, 2, 0)) # e.g. (3, 84, 84) -> (84, 84, 3) bc the latter is needed to save gif
                self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            if self.view == 'both':
                path = str(self.save_dir / file_name) + '-view_1.gif'
                imageio.mimsave(path, self.frames1, fps=self.fps)
                path = str(self.save_dir / file_name) + '-view_3.gif'
                imageio.mimsave(path, self.frames3, fps=self.fps)
            else:
                path = str(self.save_dir / file_name) + '.gif'
                imageio.mimsave(path, self.frames, fps=self.fps)


class TrainVideoRecorder:
    def __init__(self, view, root_dir, render_size=256, fps=30):
        self.view = view
        if root_dir is not None:
            self.save_dir = root_dir / 'train_video'
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        if self.view == 'both':
            self.frames1 = []
            self.frames3 = []
        else:
            self.frames = []

    def init(self, obs, enabled=True):
        self.enabled = self.save_dir is not None and enabled
        if self.view == 'both':
            self.frames1 = []
            self.frames3 = []
        else:
            self.frames = []
        self.record(obs)

    def record(self, obs):
        if self.enabled:
            if self.view == 'both':
                img_obs1, img_obs3, _ = obs
                frame1 = cv2.resize(img_obs1[-3:].transpose(1, 2, 0),
                                    dsize=(self.render_size, self.render_size),
                                    interpolation=cv2.INTER_CUBIC)
                self.frames1.append(frame1)
                frame3 = cv2.resize(img_obs3[-3:].transpose(1, 2, 0),
                                    dsize=(self.render_size, self.render_size),
                                    interpolation=cv2.INTER_CUBIC)
                self.frames3.append(frame3)
            else:
                img_obs, _ = obs
                frame = cv2.resize(img_obs[-3:].transpose(1, 2, 0),
                                   dsize=(self.render_size, self.render_size),
                                   interpolation=cv2.INTER_CUBIC)
                self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            if self.view == 'both':
                path = str(self.save_dir / file_name) + '-view_1.gif'
                imageio.mimsave(path, self.frames1, fps=self.fps)
                path = str(self.save_dir / file_name) + '-view_3.gif'
                imageio.mimsave(path, self.frames3, fps=self.fps)
            else:
                path = str(self.save_dir / file_name) + '.gif'
                imageio.mimsave(path, self.frames, fps=self.fps)
