import threading
from unittest.mock import Mock, patch

import pytest

from expressive_furhat.background_gestures import BackgroundGesturesThread
from expressive_furhat.gesture_generation import GestureGeneration


class TestGestureGeneration:
    """Test suite for GestureGeneration."""

    @pytest.fixture
    def mock_bg_thread(self, mock_furhat):
        """Mock BackgroundGesturesThread."""
        thread = Mock(spec=BackgroundGesturesThread)
        thread.add_gesture_to_cache = Mock()
        thread.flush_cache = Mock()
        thread.start = Mock()
        return thread

    @pytest.fixture
    def gesture_gen(self, mock_openai, mock_furhat, mock_bg_thread):
        """Create GestureGeneration instance with mocked dependencies."""
        return GestureGeneration(
            openai=mock_openai, furhat=mock_furhat, bg_thread=mock_bg_thread
        )

    def test_initialization_default(self, mock_openai, mock_furhat):
        """Test initialization with default parameters."""
        gen = GestureGeneration(openai=mock_openai, furhat=mock_furhat)

        assert gen.openai is mock_openai
        assert gen.furhat is mock_furhat
        assert gen.bg_thread is not None
        assert gen.mood_change_model == "gpt-4o-mini"
        assert gen.fast_generate_model == "gpt-4o-mini"
        assert gen.reasoning_model == "gpt-4o"
        assert gen.multiple_generate_model == "gpt-4o"
        assert gen._since_last_mood_change == 0

    def test_initialization_custom(self, mock_openai, mock_furhat, mock_bg_thread):
        """Test initialization with custom parameters."""
        gen = GestureGeneration(
            openai=mock_openai,
            furhat=mock_furhat,
            bg_thread=mock_bg_thread,
            mood_change_model="custom-model-1",
            fast_generate_model="custom-model-2",
            reasoning_model="custom-model-3",
            multiple_generate_model="custom-model-4",
        )

        assert gen.mood_change_model == "custom-model-1"
        assert gen.fast_generate_model == "custom-model-2"
        assert gen.reasoning_model == "custom-model-3"
        assert gen.multiple_generate_model == "custom-model-4"

    def test_on_user_input_spawns_threads(self, gesture_gen, conversation_string):
        """Test that on_user_input spawns background threads."""
        with patch("threading.Thread") as mock_thread_class:
            gesture_gen.on_user_input(conversation_string)

            # Should spawn 2 threads (multiple_generate and check_mood_change)
            assert mock_thread_class.call_count == 2

    def test_check_mood_change_no_change(
        self, gesture_gen, mock_openai, conversation_string
    ):
        """Test mood change detection when no change detected."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "0"
        mock_openai.chat.completions.create = Mock(return_value=mock_response)

        initial_count = gesture_gen._since_last_mood_change
        gesture_gen._check_mood_change(conversation_string)

        # Counter should increment
        assert gesture_gen._since_last_mood_change == initial_count + 1

        # Cache should not be flushed
        gesture_gen.bg_thread.flush_cache.assert_not_called()

    def test_check_mood_change_detected(
        self, gesture_gen, mock_openai, conversation_string
    ):
        """Test mood change detection when change is detected."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "1"
        mock_openai.chat.completions.create = Mock(return_value=mock_response)

        gesture_gen._since_last_mood_change = 5

        with patch("threading.Thread") as mock_thread:
            gesture_gen._check_mood_change(conversation_string)

        # Counter should reset
        assert gesture_gen._since_last_mood_change == 0

        # Cache should be flushed
        gesture_gen.bg_thread.flush_cache.assert_called_once()

        # Fast generate thread should be spawned
        mock_thread.assert_called_once()

    def test_fast_generate_gesture(self, gesture_gen, mock_openai, conversation_string):
        """Test fast gesture generation."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[
            0
        ].message.content = """Step 1: User is happy
Step 2: 
{
    "frames": [
        {
            "time": [0.0, 2.0],
            "params": {
                "SMILE_OPEN": 0.9,
                "BROW_UP_LEFT": 1.0
            }
        },
        {
            "time": [2.5],
            "params": {"reset": True}
        }
    ],
    "name": "Happy",
    "class": "furhatos.gestures.Gesture"
}"""
        mock_openai.chat.completions.create = Mock(return_value=mock_response)

        gesture_gen._fast_generate_gesture(conversation_string)

        # Verify gesture was added to cache
        gesture_gen.bg_thread.add_gesture_to_cache.assert_called_once()

        # Verify the parsed gesture structure
        call_args = gesture_gen.bg_thread.add_gesture_to_cache.call_args
        weight, gesture = call_args[0]

        assert isinstance(weight, int)
        assert "frames" in gesture
        assert "name" in gesture

    def test_get_robot_behavior(self, gesture_gen, mock_openai, conversation_string):
        """Test robot behavior reasoning."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[
            0
        ].message.content = """[Start reasoning]
The user is happy
[End reasoning]
1. Smile warmly
2. Nod head
3. Show interest"""
        mock_openai.chat.completions.create = Mock(return_value=mock_response)

        behavior = gesture_gen._get_robot_behavior(conversation_string)

        # Verify reasoning is extracted correctly
        assert "Smile warmly" in behavior
        assert "Nod head" in behavior
        assert "[Start reasoning]" not in behavior
        assert "[End reasoning]" not in behavior

    def test_parse_gesture_smile_scaling(self, gesture_gen):
        """Test that parse_gesture scales SMILE_OPEN and SMILE_CLOSED."""
        gestures = [
            {
                "frames": [
                    {
                        "time": [0.0, 2.0],
                        "params": {"SMILE_OPEN": 1.0, "SMILE_CLOSED": 0.5},
                    }
                ],
                "class": "furhatos.gestures.Gesture",
            }
        ]

        parsed = gesture_gen._parse_gesture(gestures)

        # SMILE_OPEN should be scaled to 0.4 (1.0 * 0.4)
        assert parsed[0]["frames"][0]["params"]["SMILE_OPEN"] == 0.4
        # SMILE_CLOSED should be scaled to 0.2 (0.5 * 0.4)
        assert parsed[0]["frames"][0]["params"]["SMILE_CLOSED"] == 0.2

    def test_parse_gesture_surprise_adds_reset(self, gesture_gen):
        """Test that SURPRISE parameter triggers reset frame addition."""
        gestures = [
            {
                "frames": [
                    {
                        "time": [0.0, 2.0],
                        "params": {"SURPRISE": 0.8, "BROW_UP_LEFT": 1.0},
                    }
                ],
                "class": "furhatos.gestures.Gesture",
            }
        ]

        parsed = gesture_gen._parse_gesture(gestures)

        # Should add reset frame at the end
        assert len(parsed[0]["frames"]) == 2
        assert parsed[0]["frames"][-1]["params"]["reset"] is True

    def test_multiple_generate_gesture_streaming(
        self, gesture_gen, mock_openai, conversation_string
    ):
        """Test multiple gesture generation with streaming."""
        # Mock streaming response
        mock_chunk1 = Mock()
        mock_chunk1.choices = [Mock()]
        mock_chunk1.choices[0].delta.content = (
            '{"frames": [{"time": [0.0, 1.0], "params": {"SMILE_OPEN": 0.5}}], '
        )

        mock_chunk2 = Mock()
        mock_chunk2.choices = [Mock()]
        mock_chunk2.choices[0].delta.content = '"class": "furhatos.gestures.Gesture"}'

        mock_openai.chat.completions.create = Mock(
            return_value=[mock_chunk1, mock_chunk2]
        )

        # Mock the reasoning step
        with patch.object(
            gesture_gen, "_get_robot_behavior", return_value="Test behavior"
        ):
            # Set cache ready immediately for testing
            gesture_gen._cache_ready.set()

            gesture_gen._multiple_generate_gesture(conversation_string)

        # Verify at least one gesture was added to cache
        assert gesture_gen.bg_thread.add_gesture_to_cache.call_count >= 1

    def test_thread_safety_since_last_mood_change(self, gesture_gen):
        """Test thread-safe access to _since_last_mood_change."""

        def increment():
            for _ in range(100):
                with gesture_gen._lock:
                    current = gesture_gen._since_last_mood_change
                    gesture_gen._since_last_mood_change = current + 1

        threads = [threading.Thread(target=increment) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should be 500 (100 * 5 threads)
        assert gesture_gen._since_last_mood_change == 500

    def test_cache_ready_event(self, gesture_gen, mock_openai, conversation_string):
        """Test that _cache_ready event is used correctly."""
        gesture_gen._cache_ready.set()

        # Mock responses
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "0"
        mock_openai.chat.completions.create = Mock(return_value=mock_response)

        # Clear and check mood change
        gesture_gen._cache_ready.clear()
        gesture_gen._check_mood_change(conversation_string)

        # Event should be set after check
        assert gesture_gen._cache_ready.is_set()

    def test_openai_api_calls_mood_change(
        self, gesture_gen, mock_openai, conversation_string
    ):
        """Test OpenAI API is called with correct parameters for mood change."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "0"
        mock_openai.chat.completions.create = Mock(return_value=mock_response)

        gesture_gen._check_mood_change(conversation_string)

        # Verify API call
        mock_openai.chat.completions.create.assert_called_once()
        call_kwargs = mock_openai.chat.completions.create.call_args[1]

        assert call_kwargs["model"] == gesture_gen.mood_change_model
        assert call_kwargs["temperature"] == 0.0
        assert "messages" in call_kwargs

    def test_parse_gesture_no_reset_frame(self, gesture_gen):
        """Test parsing gesture without reset frame."""
        gestures = [
            {
                "frames": [
                    {
                        "time": [0.0, 2.0],
                        "params": {"NECK_TILT": -5.0, "BROW_UP_LEFT": 0.5},
                    }
                ],
                "class": "furhatos.gestures.Gesture",
            }
        ]

        parsed = gesture_gen._parse_gesture(gestures)

        # Should not add reset frame (no illegal emotions)
        assert len(parsed[0]["frames"]) == 1
        assert "reset" not in parsed[0]["frames"][0]["params"]
