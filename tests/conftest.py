from contextlib import contextmanager
from unittest.mock import Mock, patch

import pytest
from openai import OpenAI


@pytest.fixture
def mock_furhat():
    """Mock FurhatRemoteAPI instance."""
    mock = Mock()
    mock.gesture = Mock(return_value={"success": True})
    mock.attend = Mock(return_value={"success": True})
    mock.say = Mock(return_value={"success": True})
    mock.set_face = Mock(return_value={"success": True})
    mock.set_voice = Mock(return_value={"success": True})
    mock.set_led = Mock(return_value={"success": True})
    return mock


@pytest.fixture
def mock_openai():
    """Mock OpenAI client instance."""
    mock = Mock(spec=OpenAI)

    # Mock the completions structure
    mock.chat = Mock()
    mock.chat.completions = Mock()

    return mock


@pytest.fixture
def sample_gesture():
    """Sample gesture dictionary for testing."""
    return {
        "frames": [
            {
                "time": [0.0, 2.0],
                "params": {
                    "SMILE_OPEN": 0.9,
                    "BROW_UP_LEFT": 1.0,
                    "BROW_UP_RIGHT": 1.0,
                    "NECK_TILT": -10.0,
                },
            },
            {"time": [2.5], "params": {"reset": True}},
        ],
        "name": "Happy Gesture",
        "class": "furhatos.gestures.Gesture",
    }


@pytest.fixture
def sample_conversation():
    """Sample conversation for testing."""
    return [
        {"role": "user", "content": "Hello!"},
        {"role": "Furhat", "content": "Hi there! How can I help you today?"},
        {"role": "user", "content": "I'm feeling great!"},
    ]


@pytest.fixture
def conversation_string(sample_conversation):
    """Convert sample conversation to string format."""
    return "\n".join(
        [f"{item['role']}: {item['content']}" for item in sample_conversation]
    )


@contextmanager
def patch_furhat_init():
    """Context manager to properly patch FurhatRemoteAPI initialization.

    This patches the parent class __init__ and the methods called during
    ExpressiveRemoteAPI initialization (set_face, set_voice, set_led).
    """
    with patch(
        "expressive_furhat.expressive_remote_api.FurhatRemoteAPI.__init__",
        return_value=None,
    ):
        with patch(
            "expressive_furhat.expressive_remote_api.FurhatRemoteAPI.set_face"
        ), patch(
            "expressive_furhat.expressive_remote_api.FurhatRemoteAPI.set_voice"
        ), patch(
            "expressive_furhat.expressive_remote_api.FurhatRemoteAPI.set_led"
        ):
            yield
