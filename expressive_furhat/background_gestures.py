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

DEFAULT_WEIGHTING_FACTOR = 1.4
DEFAULT_WAIT_SECONDS = 1.0


class BackgroundGesturesThread(threading.Thread):
    """Thread to display background gestures.

    When started, with `start()`, the thread will periodically sample from its
    internal cache a random gesture and display it on the Furhat robot. The
    gestures are  displayed every `self.wait_seconds`. The cycle will repeat
    until `join()` is called, which will stop the thread immediately.

    You can pause and resume the gesture execution by calling `pause()` and
    `resume()` methods. You can add gestures to the internal cache by calling
    the `add_gesture_to_cache` method, providing a weighting factor and the
    gesture dictionary. To reset the cache, call the `flush_cache` method.

    The sampling method is geometric with a weighting factor of
    `self.weighing_factor`. For instance, an element inserted with
    `add_gesture_to_cache(x, element)` will have a weight of
    `self.weighing_factor**x` during sampling.
    """

    furhat: FurhatRemoteAPI
    weighting_factor: float
    wait_seconds: float

    _gestures_cache: list[tuple[int, dict]]
    _is_stopped: threading.Event
    _is_paused: threading.Event
    _lock: threading.Lock
    _logger: logging.Logger

    def __init__(
        self,
        furhat: FurhatRemoteAPI,
        weighting_factor: float = DEFAULT_WEIGHTING_FACTOR,
        wait_seconds: float = DEFAULT_WAIT_SECONDS,
    ):
        """Initialize the background gestures thread.

        Args:
            furhat (FurhatRemoteAPI): The furhat API instance to use.
            weighting_factor (float, optional): The weighting factor for the
                exponential sampling. Defaults to `DEFAULT_WEIGHTING_FACTOR`.
            wait_seconds (float, optional): The number of seconds to wait
                between gestures. Defaults to `DEFAULT_WAIT_SECONDS`.
        """
        super().__init__(daemon=True)
        self.furhat = furhat
        self.weighting_factor = weighting_factor
        self.wait_seconds = wait_seconds
        
        self._gestures_cache = []
        self._is_stopped = threading.Event()
        self._is_paused = threading.Event()
        self._lock = threading.Lock()
        self._logger = logging.getLogger(__name__)

    @override
    def join(self, timeout: Optional[float] = None) -> None:
        self._is_stopped.set()
        self._logger.info("Thread stopping")
        return super().join(timeout)

    @override
    def run(self) -> None:
        """Do not call this method directly, use `start` instead."""
        self._logger.info("Thread started")
        while not self._is_stopped.is_set():
            if self._is_paused.is_set():
                self._is_stopped.wait(self.wait_seconds)
                continue
            gesture = self._sample_from_cache()
            if gesture is not None:
                self._send_gesture_blocking(gesture)
            self._is_stopped.wait(self.wait_seconds)
        self._logger.info("Thread stopped")

    def add_gesture_to_cache(self, weight: int, gesture: dict) -> None:
        """Add a gesture to the cache with a given weighting factor."""
        with self._lock:
            self._gestures_cache.append((weight, gesture))

    def flush_cache(self) -> None:
        """Clear the gesture cache."""
        with self._lock:
            self._gestures_cache.clear()

    def pause(self) -> None:
        """Pause the gesture execution."""
        self._is_paused.set()

    def resume(self) -> None:
        """Resume the gesture execution."""
        self._is_paused.clear()

    def _sample_from_cache(self) -> Optional[dict]:
        with self._lock:
            if not self._gestures_cache:
                return None
            weights, gestures = zip(
                *((self.weighting_factor**i, g) for i, g in self._gestures_cache)
            )
        return random.choices(gestures, weights=weights, k=1)[0]

    def _send_gesture_blocking(self, gesture: dict) -> None:
        self.furhat.gesture(
            body=gesture,
            blocking=True,
        )
