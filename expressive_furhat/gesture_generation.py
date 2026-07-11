# Copyright 2025 Giulio Antonio Abbo
#
# Parts of this file are adapted from: Van de Vreken, S. (2025). Generating
# Facial Expressions for Social Robots using Large Language Models [Master’s
# thesis, Ghent University].
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

import ast
import logging
import re
import threading
from typing import cast

from furhat_remote_api import FurhatRemoteAPI
from openai import OpenAI

from expressive_furhat.background_gestures import BackgroundGesturesThread
from expressive_furhat.prompts import (
    FAST_GENERATE_SP,
    MOOD_CHANGE_SP,
    MULTIPLE_GENERATE_SP,
    MULTIPLE_GENERATE_UP,
    REASONING_SP,
    REASONING_UP,
)


class GestureGeneration:
    """Generate gestures for Furhat based on the ongoing conversation.

    To generate gestures, this class uses multiple OpenAI models to:

    1. Detect mood changes in the conversation.
    2. Quickly generate a gesture when a mood change is detected.
    3. Reason about the robot's behavior based on the conversation.
    4. Generate multiple gestures based on the reasoned behavior.

    To use this class, create an instance providing the OpenAI API and Furhat
    Remote API instances. To generate gestures based on user input, call the
    `on_user_input` method with the conversation history as a string.
    """

    bg_thread: BackgroundGesturesThread

    openai: OpenAI
    furhat: FurhatRemoteAPI
    mood_change_model: str
    fast_generate_model: str
    reasoning_model: str
    multiple_generate_model: str
    mood_change_system_prompt: str
    fast_generate_system_prompt: str
    reasoning_system_prompt: str
    reasoning_user_prompt: str
    multiple_generate_system_prompt: str
    multiple_generate_user_prompt: str

    _since_last_mood_change: int
    _lock: threading.Lock
    _cache_ready: threading.Event
    _logger: logging.Logger

    def __init__(
        self,
        openai: OpenAI,
        furhat: FurhatRemoteAPI,
        bg_thread: BackgroundGesturesThread | None = None,
        mood_change_model: str = "gpt-4o-mini",
        fast_generate_model: str = "gpt-4o-mini",
        reasoning_model: str = "gpt-4o",
        multiple_generate_model: str = "gpt-4o",
        mood_change_system_prompt: str = MOOD_CHANGE_SP,
        fast_generate_system_prompt: str = FAST_GENERATE_SP,
        reasoning_system_prompt: str = REASONING_SP,
        reasoning_user_prompt: str = REASONING_UP,
        multiple_generate_system_prompt: str = MULTIPLE_GENERATE_SP,
        multiple_generate_user_prompt: str = MULTIPLE_GENERATE_UP,
    ):
        """Initialize the gesture generation.

        Args:
            openai (OpenAI): The OpenAI API instance to use to generate the
                gestures (it can be an Ollama instance).
            furhat (FurhatRemoteAPI): The Furhat API instance to use to display
                the gestures.
            bg_thread (BackgroundGesturesThread, optional): The background
                thread to use, created internally if not provided.
            mood_change_model (str, optional): The model to use for mood change
                detection. Defaults to "gpt-4o-mini".
            fast_generate_model (str, optional): The model to use for fast
                gesture generation. Defaults to "gpt-4o-mini".
            reasoning_model (str, optional): The model to use for reasoning on
                robot behavior. Defaults to "gpt-4o".
            multiple_generate_model (str, optional): The model to use for
                multiple gesture generation. Defaults to "gpt-4o".
            mood_change_system_prompt (str, optional): The system prompt for
                mood change detection. Defaults to `MOOD_CHANGE_SP`.
            fast_generate_system_prompt (str, optional): The system prompt for
                fast gesture generation. Defaults to `FAST_GENERATE_SP`.
            reasoning_system_prompt (str, optional): The system prompt for
                reasoning on robot behavior. Defaults to `REASONING_SP`.
            reasoning_user_prompt (str, optional): The user prompt for reasoning
                on robot behavior. Defaults to `REASONING_UP`.
            multiple_generate_system_prompt (str, optional): The system prompt
                for multiple gesture generation. Defaults to
                `MULTIPLE_GENERATE_SP`.
            multiple_generate_user_prompt (str, optional): The user prompt for
                multiple gesture generation. Defaults to `MULTIPLE_GENERATE_UP`.
        """
        self.openai = openai
        self.furhat = furhat
        self.bg_thread = bg_thread or BackgroundGesturesThread(self.furhat)
        # Only start the thread if it's not already running
        if not self.bg_thread.is_alive():
            self.bg_thread.start()
        self.mood_change_model = mood_change_model
        self.fast_generate_model = fast_generate_model
        self.reasoning_model = reasoning_model
        self.multiple_generate_model = multiple_generate_model
        self.mood_change_system_prompt = mood_change_system_prompt
        self.fast_generate_system_prompt = fast_generate_system_prompt
        self.reasoning_system_prompt = reasoning_system_prompt
        self.reasoning_user_prompt = reasoning_user_prompt
        self.multiple_generate_system_prompt = multiple_generate_system_prompt
        self.multiple_generate_user_prompt = multiple_generate_user_prompt

        self._since_last_mood_change = 0
        self._lock = threading.Lock()
        self._cache_ready = threading.Event()
        self._logger = logging.getLogger(__name__)

    def on_user_input(self, conversation: str):
        """Generate a gesture based on the user's input.

        This method will start a thread to generate a gesture based on the
        user's input, which will first reason on the robot's behavior and then
        generate a series of gestures based on that. It will also start a thread
        to check for mood changes and flush the cache if a mood change is
        detected. A gesture will be quickly generated if a mood change is
        detected.

        Args:
            conversation (str): A string representing the conversation between
                the user and the robot.
        """
        self.on_user_input_with_emotion(conversation=conversation, emotion=None)

    def on_user_input_with_emotion(
        self, conversation: str, emotion: str | None = None
    ):
        """Generate gestures with an optional forced emotion.

        If ``emotion`` is ``None`` (or the string ``"none"``), behavior is
        identical to ``on_user_input``. Otherwise, the emotion is injected into
        prompt messages to strongly bias/force generation toward that emotion.

        Args:
            conversation (str): A string representing the conversation between
                the user and the robot.
            emotion (str | None, optional): Emotion to force in generated
                gestures (e.g. ``"joy"``, ``"sadness"``, ``"anger"``).
                Defaults to ``None``.
        """
        normalized_emotion = self._normalize_emotion(emotion)
        self._cache_ready.clear()
        threading.Thread(
            target=self._multiple_generate_gesture,
            args=(conversation, normalized_emotion),
            daemon=True,
        ).start()

        threading.Thread(
            target=self._check_mood_change,
            args=(conversation, normalized_emotion),
            daemon=True,
        ).start()

    def _check_mood_change(self, conversation: str, emotion: str | None = None):
        messages = [
            {"role": "system", "content": self.mood_change_system_prompt},
            {"role": "user", "content": conversation},
        ]
        response = self.openai.chat.completions.create(
            model=self.mood_change_model,
            messages=messages,  # type: ignore
            temperature=0.0,
        )
        mood_change = cast(str, response.choices[0].message.content).strip()

        if mood_change == "1":
            self._logger.info("Mood change detected")
            with self._lock:
                self._since_last_mood_change = 0
            self.bg_thread.flush_cache()
            threading.Thread(
                target=self._fast_generate_gesture,
                args=(conversation, emotion),
                daemon=True,
            ).start()
        else:
            self._logger.info("No mood change detected, value: %s", mood_change)
            with self._lock:
                self._since_last_mood_change += 1
        self._cache_ready.set()

    def _fast_generate_gesture(self, conversation: str, emotion: str | None = None):
        emotion_instruction = self._emotion_instruction(emotion)
        messages = [
            {
                "role": "system",
                "content": self.fast_generate_system_prompt + emotion_instruction,
            },
            {"role": "user", "content": conversation + emotion_instruction},
        ]
        response = self.openai.chat.completions.create(
            model=self.fast_generate_model,
            messages=messages,  # type: ignore
            temperature=0.0,
        )
        gesture = cast(str, response.choices[0].message.content).strip()

        # extract everything after 'Step 2:'
        match = re.search(r"Step 2:\s*(.*)", gesture, re.DOTALL)
        if match:
            gesture = match.group(1).strip()

        # Convert the string to a Python dictionary
        gesture = ast.literal_eval(gesture)
        gesture = self._parse_gesture([gesture])[0]
        with self._lock:
            self.bg_thread.add_gesture_to_cache(self._since_last_mood_change, gesture)

    def _multiple_generate_gesture(
        self, conversation: str, emotion: str | None = None
    ):
        emotion_instruction = self._emotion_instruction(emotion)
        robot_behavior = self._get_robot_behavior(conversation, emotion)
        structured_conversation = self.multiple_generate_user_prompt.format(
            conversation=conversation, robot_behavior=robot_behavior
        )
        messages = [
            {
                "role": "system",
                "content": self.multiple_generate_system_prompt + emotion_instruction,
            },
            {
                "role": "user",
                "content": structured_conversation + emotion_instruction,
            },
        ]
        response_stream = self.openai.chat.completions.create(
            model=self.multiple_generate_model,
            messages=messages,  # type: ignore
            temperature=0.0,
            stream=True,
        )

        # wait until there is no mood change or the cache is flushed
        self._cache_ready.wait()

        gesture_buffer = ""
        open_braces = 0
        for chunk in response_stream:
            delta = chunk.choices[0].delta.content or ""
            for char in delta:
                if char == "{":
                    open_braces += 1
                if open_braces > 0:
                    gesture_buffer += char
                if char == "}":
                    open_braces -= 1
                    if open_braces == 0:
                        try:
                            gesture_data = ast.literal_eval(gesture_buffer)
                            if (
                                isinstance(gesture_data, dict)
                                and "frames" in gesture_data
                            ):
                                updated_gesture = self._parse_gesture([gesture_data])[0]
                                with self._lock:
                                    self.bg_thread.add_gesture_to_cache(
                                        self._since_last_mood_change, updated_gesture
                                    )
                            else:
                                self._logger.warning(
                                    f"Invalid gesture structure: {gesture_data}"
                                )
                        except (SyntaxError, ValueError) as e:
                            self._logger.warning(
                                f"Error parsing streamed gesture: {e}\nPartial buffer: {gesture_buffer}"
                            )
                        gesture_buffer = ""

    def _get_robot_behavior(self, conversation: str, emotion: str | None = None):
        emotion_instruction = self._emotion_instruction(emotion)
        prompt = self.reasoning_user_prompt.format(conversation=conversation)

        messages = [
            {
                "role": "system",
                "content": self.reasoning_system_prompt + emotion_instruction,
            },
            {"role": "user", "content": prompt + emotion_instruction},
        ]
        response = self.openai.chat.completions.create(
            model=self.reasoning_model,
            messages=messages,  # type: ignore
            temperature=0.0,
        )
        robot_behavior = cast(str, response.choices[0].message.content).strip()

        # Try to extract the behavior after [End reasoning]
        if "[End reasoning]" in robot_behavior:
            robot_behavior = robot_behavior.split("[End reasoning]")[1].strip()
        else:
            self._logger.warning(
                "Response did not contain [End reasoning] marker, using full response"
            )
        return robot_behavior

    def _normalize_emotion(self, emotion: str | None) -> str | None:
        if emotion is None:
            return None
        normalized = emotion.strip()
        if normalized == "" or normalized.lower() in {"none", "null", "default"}:
            return None
        return normalized

    def _emotion_instruction(self, emotion: str | None) -> str:
        normalized = self._normalize_emotion(emotion)
        if normalized is None:
            return ""
        return (
            "\n\n[Emotion constraint]\n"
            f"Target emotion: {normalized}.\n"
            "You MUST make generated behavior and gestures clearly express this "
            "emotion. Keep consistency with this target emotion across all "
            "generated gestures."
        )

    def _parse_gesture(
        self, list_of_gestures: list[dict[str, float]]
    ) -> list[dict[str, float]]:
        # rescale SMILE_OPEN and SMILE_CLOSED to -0.3, 0.3 during speech and add
        # 'reset frame' to end each gesture
        for i in list_of_gestures:
            frames = i["frames"]
            illegal_emotions_present = False
            for frame in frames:  # Iterate through all frames except the last one
                if "SMILE_OPEN" in frame["params"]:
                    frame["params"]["SMILE_OPEN"] = round(
                        frame["params"]["SMILE_OPEN"] * 0.4, 1
                    )
                if "SMILE_CLOSED" in frame["params"]:
                    frame["params"]["SMILE_CLOSED"] = round(
                        frame["params"]["SMILE_CLOSED"] * 0.4, 1
                    )
                if "SURPRISE" in frame["params"]:
                    illegal_emotions_present = True
            if illegal_emotions_present:
                reset_time = (
                    frames[-1]["time"]
                    if isinstance(frames[-1]["time"], float)
                    else frames[-1]["time"][-1]
                )
                frames.append({"time": [reset_time * 1.3], "params": {"reset": True}})

        return list_of_gestures
