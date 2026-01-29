# Copyright 2025 Giulio Antonio Abbo
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

import logging
import random
import threading
from typing import Optional, override

from furhat_remote_api import FurhatRemoteAPI

DEFAULT_ENGAGED_DURATION_RANGE = [2.0, 5.0]
DEFAULT_EVERSION_DURATION_RANGE = [0.5, 4.0]
DEFAULT_PAN_RANGE = [-20, 20]
DEFAULT_TILT_RANGE = [-30, 30]
DEFAULT_EYE_MOVEMENT_SPEED = 0.2


class GazeAversionThread(threading.Thread):
    """Thread to control gaze aversion.

    When started, the robot will periodically break eye contact with the user.
    After a random period of time, based on `self.engaged_duration_range`, the
    robot will engage in a gaze aversion. The eye movement will last
    `self.eye_movement_speed` seconds, towards a random pan and tilt, based on
    `self.pan_range` and `self.tilt_range`. The robot will gaze in that
    direction for a random period of time, based on
    `self.eversion_duration_range`. After that, the robot will look in the
    original direction and attend the closest user. The cycle will repeat until
    the thread is stopped calling `join()`. Stopping the thread will stop the
    gaze aversion immediately and attend the closest user.

    Gaze aversion uses `GAZE_PAN` and `GAZE_TILT` as in
    https://docs.furhat.io/remote-api/#gesturedefinition, in order to move the
    eye without moving the head.

    You can customise the actions to perform gaze aversion and eye contact by
    overriding the `gaze_aversion_resumed()` and `gaze_aversion_paused()`
    methods. These methods must be non blocking.
    """

    furhat: FurhatRemoteAPI
    engaged_duration_range: list[float]
    eversion_duration_range: list[float]
    pan_range: list[int]
    tilt_range: list[int]
    eye_movement_speed: float

    _is_stopped: threading.Event
    _logger: logging.Logger

    def __init__(
        self,
        furhat: FurhatRemoteAPI,
        engaged_duration_range=DEFAULT_ENGAGED_DURATION_RANGE,
        eversion_duration_range=DEFAULT_EVERSION_DURATION_RANGE,
        pan_range=DEFAULT_PAN_RANGE,
        tilt_range=DEFAULT_TILT_RANGE,
        eye_movement_speed=DEFAULT_EYE_MOVEMENT_SPEED,
    ):
        """Initialize the gaze aversion thread.

        Args:
            furhat (FurhatRemoteAPI): The furhat API instance to use.
            engaged_duration_range (list, optional): Duration range of the eye
                contact in seconds. Defaults to
                `DEFAULT_ENGAGED_DURATION_RANGE`.
            eversion_duration_range (list, optional): Duration range of the gaze
                aversion in seconds. Defaults to
                `DEFAULT_EVERSION_DURATION_RANGE`.
            pan_range (list, optional): Pan range of the gaze aversion as in
                https://docs.furhat.io/remote-api/#gesturedefinition. Defaults
                to `DEFAULT_PAN_RANGE`.
            tilt_range (list, optional): Tilt range of the gaze aversion as in
                https://docs.furhat.io/remote-api/#gesturedefinition. Defaults
                to `DEFAULT_TILT_RANGE`.
            eye_movement_speed (float, optional): Time to move the eye from one
                position to the other in seconds. Defaults to
                `DEFAULT_EYE_MOVEMENT_SPEED`.
        """

        super().__init__(daemon=True)
        self.furhat = furhat
        self.engaged_duration_range = engaged_duration_range
        self.eversion_duration_range = eversion_duration_range
        self.pan_range = pan_range
        self.tilt_range = tilt_range
        self.eye_movement_speed = eye_movement_speed

        self._is_stopped = threading.Event()
        self._logger = logging.getLogger(__name__)

    @override
    def join(self, timeout: float | None = None) -> None:
        self._is_stopped.set()
        self._logger.info("Thread stopping")
        return super().join(timeout)

    @override
    def run(self) -> None:
        """Do not call this method directly, use `start` instead."""
        self._logger.info("Thread started")
        while not self._is_stopped.is_set():
            engaged_t = random.uniform(*self.engaged_duration_range)
            self._logger.info(f"Pausing gaze aversion for {engaged_t} seconds")
            self.gaze_aversion_paused(engaged_t)
            stopped = self._is_stopped.wait(engaged_t)
            if stopped:
                break

            aversion_t = random.uniform(*self.eversion_duration_range)
            self._logger.info(f"Resuming gaze aversion for {aversion_t} seconds")
            self.gaze_aversion_resumed(aversion_t)
            stopped = self._is_stopped.wait(aversion_t)
            if stopped:
                break
        self.gaze_aversion_paused()
        self._logger.info("Thread stopped")

    def gaze_aversion_resumed(self, duration: float) -> None:
        """Non blocking action to perform gaze aversion for a given duration."""
        pan = random.uniform(*self.pan_range)
        tilt = random.uniform(*self.tilt_range)
        self.furhat.gesture(
            blocking=False,
            body={
                "name": "GazeAversion",
                "class": "furhatos.gestures.Gesture",
                "frames": [
                    {
                        "time": [0.0, self.eye_movement_speed],
                        "params": {
                            "GAZE_PAN": pan,
                            "GAZE_TILT": tilt,
                        },
                    },
                    {
                        "time": [
                            self.eye_movement_speed + duration,
                            2 * self.eye_movement_speed + duration,
                        ],
                        "params": {"reset": True},
                    },
                ],
            },
        )

    def gaze_aversion_paused(self, duration: Optional[float] = None) -> None:
        """Non blocking action to engage in eye contact for a given duration."""
        self.furhat.attend(user="CLOSEST")
