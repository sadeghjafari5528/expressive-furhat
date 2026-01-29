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

import threading
from typing import override

from furhat_remote_api import FurhatRemoteAPI
from openai import OpenAI

from expressive_furhat.gaze_aversion import GazeAversionThread
from expressive_furhat.gesture_generation import GestureGeneration


class ExpressiveRemoteAPI(FurhatRemoteAPI):
    """Remote API for expressive furhat.

    This class extends the original FurhatRemoteAPI class to add gaze aversion
    while speaking and generate gestures based on the ongoing conversation.
    """

    openai: OpenAI
    gesture_gen: GestureGeneration
    with_gaze_aversion: bool

    _conversation: list[dict[str, str]]

    def __init__(
        self,
        *args,
        openai: OpenAI,
        gesture_gen: GestureGeneration | None = None,
        with_gaze_aversion: bool = True,
        mask: str = "adult",
        character: str = "default",
        voice: str = "Matthew",
        leds: dict = {"red": 0, "green": 0, "blue": 0},
        **kwargs,
    ):
        """Initialize the expressive remote API.

        Args:
            *args: Positional arguments to pass to the original constructor.
            openai (OpenAI): The OpenAI API (or Ollama) instance to use.
            gesture_gen (GestureGeneration, optional): The gesture generation
                instance to use. Defaults to None.
            with_gaze_aversion (bool, optional): Whether to add gaze aversion
                while speaking. Defaults to True.
            mask (str, optional): The mask to use for the robot. Defaults to
                "adult".
            character (str, optional): The character to use for the robot.
                Defaults to "default".
            voice (str, optional): The voice to use for the robot. Defaults to
                "Matthew".
            leds (dict, optional): The LEDs to use for the robot. Defaults to
                all LEDs off.
            **kwargs: Keyword arguments to pass to the original constructor.
        """
        super().__init__(*args, **kwargs)
        self.openai = openai
        self.gesture_gen = gesture_gen or GestureGeneration(self.openai, self)
        self.with_gaze_aversion = with_gaze_aversion

        self._conversation = []

        self.set_face(mask=mask, character=character)
        self.set_voice(name=voice)
        self.set_led(**leds)

    def react_to_text(self, text: str):
        """Track the user's input and generate gestures based on it."""
        self._conversation.append({"role": "user", "content": text})
        transcript = "\n".join(
            [f"{item['role']}: {item['content']}" for item in self._conversation]
        )
        self.gesture_gen.on_user_input(transcript)

    @override
    def say(self, **kwargs):
        """Override the say method to add gaze aversion while speaking and track
        the robot's output.

        Consult the docstring of the original method for more information.

        Note: if no text is provided, this method will not track the robot's
        output.
        """
        if "text" in kwargs:
            self._conversation.append({"role": "Furhat", "content": kwargs["text"]})

        if not self.with_gaze_aversion:
            return super().say(**kwargs)

        th = GazeAversionThread(self)
        th.start()

        if (
            "blocking" in kwargs
            and kwargs["blocking"] is True
            or kwargs["blocking"] == "True"
        ):
            result = super().say(**kwargs)
            th.join()
            return result

        kwargs["blocking"] = True

        def _say():
            super().say(**kwargs)
            th.join()

        threading.Thread(target=_say, daemon=True).start()
        return {"message": "Speaking with gaze aversion", "success": True}
