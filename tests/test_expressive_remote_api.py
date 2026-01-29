from unittest.mock import Mock, patch

import pytest

from expressive_furhat.expressive_remote_api import ExpressiveRemoteAPI
from expressive_furhat.gesture_generation import GestureGeneration
from tests.conftest import patch_furhat_init


class TestExpressiveRemoteAPI:
    """Test suite for ExpressiveRemoteAPI."""

    @pytest.fixture
    def mock_gesture_gen(self, mock_furhat):
        """Mock GestureGeneration instance."""
        gen = Mock(spec=GestureGeneration)
        gen.on_user_input = Mock()
        return gen

    @pytest.fixture
    def api(self, mock_openai, mock_gesture_gen):
        """Create ExpressiveRemoteAPI instance with mocked dependencies."""
        with patch_furhat_init():
            api = ExpressiveRemoteAPI(
                openai=mock_openai,
                gesture_gen=mock_gesture_gen,
                with_gaze_aversion=True,
            )
            # Mock the parent class methods except 'say'
            api.gesture = Mock()
            api.attend = Mock()
            return api

    def test_initialization_default(self, mock_openai):
        """Test initialization with default parameters."""
        with patch_furhat_init():
            with patch(
                "expressive_furhat.expressive_remote_api.GestureGeneration"
            ) as mock_gen_class:
                api = ExpressiveRemoteAPI(openai=mock_openai)

                assert api.openai is mock_openai
                assert api.with_gaze_aversion is True
                assert api._conversation == []

                # Verify default initialization calls
                mock_gen_class.assert_called_once()

    def test_initialization_custom(self, mock_openai, mock_gesture_gen):
        """Test initialization with custom parameters."""
        with patch_furhat_init():
            api = ExpressiveRemoteAPI(
                openai=mock_openai,
                gesture_gen=mock_gesture_gen,
                with_gaze_aversion=False,
                mask="child",
                character="custom",
                voice="Emma",
                leds={"red": 255, "green": 0, "blue": 0},
            )

            assert api.gesture_gen is mock_gesture_gen
            assert api.with_gaze_aversion is False

    def test_react_to_text(self, api, mock_gesture_gen):
        """Test reacting to user text input."""
        api._conversation = [{"role": "Furhat", "content": "Hello!"}]

        api.react_to_text("How are you?")

        # Verify conversation is updated
        assert len(api._conversation) == 2
        assert api._conversation[-1] == {"role": "user", "content": "How are you?"}

        # Verify gesture generation was triggered
        mock_gesture_gen.on_user_input.assert_called_once()
        call_arg = mock_gesture_gen.on_user_input.call_args[0][0]
        assert "Furhat: Hello!" in call_arg
        assert "user: How are you?" in call_arg

    def test_say_without_gaze_aversion(self, api):
        """Test say method when gaze aversion is disabled."""
        api.with_gaze_aversion = False

        with patch(
            "expressive_furhat.expressive_remote_api.FurhatRemoteAPI.say"
        ) as mock_parent_say:
            mock_parent_say.return_value = {"success": True}
            result = api.say(text="Hello", blocking=True)

        # Should call parent's say method directly
        mock_parent_say.assert_called_once_with(text="Hello", blocking=True)
        assert result == {"success": True}

    def test_say_with_text_updates_conversation(self, api):
        """Test that say with text updates conversation history."""
        with patch("expressive_furhat.expressive_remote_api.FurhatRemoteAPI.say"):
            with patch(
                "expressive_furhat.expressive_remote_api.GazeAversionThread"
            ) as mock_gaze:
                mock_thread = Mock()
                mock_thread.start = Mock()
                mock_thread.join = Mock()
                mock_gaze.return_value = mock_thread

                api.say(text="Hello there!", blocking=True)

        # Verify conversation is updated
        assert len(api._conversation) == 1
        assert api._conversation[0] == {"role": "Furhat", "content": "Hello there!"}

    def test_say_without_text_no_conversation_update(self, api):
        """Test that say without text does not update conversation."""
        with patch("expressive_furhat.expressive_remote_api.FurhatRemoteAPI.say"):
            with patch(
                "expressive_furhat.expressive_remote_api.GazeAversionThread"
            ) as mock_gaze:
                mock_thread = Mock()
                mock_thread.start = Mock()
                mock_thread.join = Mock()
                mock_gaze.return_value = mock_thread

                api.say(url="http://example.com/audio.wav", blocking=True)

        # Conversation should not be updated
        assert len(api._conversation) == 0

    def test_say_blocking_waits_for_gaze_thread(self, api):
        """Test that blocking say waits for gaze aversion thread."""
        with patch(
            "expressive_furhat.expressive_remote_api.FurhatRemoteAPI.say"
        ) as mock_parent_say:
            with patch(
                "expressive_furhat.expressive_remote_api.GazeAversionThread"
            ) as mock_gaze:
                mock_thread = Mock()
                mock_thread.start = Mock()
                mock_thread.join = Mock()
                mock_gaze.return_value = mock_thread

                mock_parent_say.return_value = {"success": True}

                result = api.say(text="Hello", blocking=True)

                # Verify thread lifecycle
                mock_thread.start.assert_called_once()
                mock_thread.join.assert_called_once()

                # Verify parent say was called
                mock_parent_say.assert_called_once()

                assert result == {"success": True}

    def test_say_non_blocking_spawns_wrapper_thread(self, api):
        """Test that non-blocking say spawns a wrapper thread."""
        with patch(
            "expressive_furhat.expressive_remote_api.FurhatRemoteAPI.say"
        ) as mock_parent_say:
            with patch(
                "expressive_furhat.expressive_remote_api.GazeAversionThread"
            ) as mock_gaze_class:
                with patch("threading.Thread") as mock_thread_class:
                    mock_gaze_thread = Mock()
                    mock_gaze_thread.start = Mock()
                    mock_gaze_thread.join = Mock()
                    mock_gaze_class.return_value = mock_gaze_thread

                    mock_wrapper_thread = Mock()
                    mock_thread_class.return_value = mock_wrapper_thread

                    result = api.say(text="Hello", blocking=False)

                    # Should spawn wrapper thread
                    mock_thread_class.assert_called_once()
                    call_kwargs = mock_thread_class.call_args[1]
                    assert call_kwargs["daemon"] is True
                    mock_wrapper_thread.start.assert_called_once()

                    # Should return immediately with success message
                    assert result["success"] is True
                    assert "Speaking with gaze aversion" in result["message"]

    def test_say_blocking_true_string(self, api):
        """Test that blocking="True" string is handled correctly."""
        with patch(
            "expressive_furhat.expressive_remote_api.FurhatRemoteAPI.say"
        ) as mock_parent_say:
            with patch(
                "expressive_furhat.expressive_remote_api.GazeAversionThread"
            ) as mock_gaze:
                mock_thread = Mock()
                mock_thread.start = Mock()
                mock_thread.join = Mock()
                mock_gaze.return_value = mock_thread

                mock_parent_say.return_value = {"success": True}

                result = api.say(text="Hello", blocking="True")

                # Should wait for thread
                mock_thread.join.assert_called_once()

    def test_conversation_tracking_multiple_turns(self, api, mock_gesture_gen):
        """Test conversation tracking over multiple turns."""
        with patch("expressive_furhat.expressive_remote_api.FurhatRemoteAPI.say"):
            with patch(
                "expressive_furhat.expressive_remote_api.GazeAversionThread"
            ) as mock_gaze:
                mock_thread = Mock()
                mock_thread.start = Mock()
                mock_thread.join = Mock()
                mock_gaze.return_value = mock_thread

                # Turn 1
                api.say(text="Hello!", blocking=True)
                api.react_to_text("Hi there!")

                # Turn 2
                api.say(text="How can I help?", blocking=True)
                api.react_to_text("I need assistance")

                # Verify conversation history
                assert len(api._conversation) == 4
                assert api._conversation[0]["role"] == "Furhat"
                assert api._conversation[1]["role"] == "user"
                assert api._conversation[2]["role"] == "Furhat"
                assert api._conversation[3]["role"] == "user"

    def test_gaze_aversion_thread_created_per_say(self, api):
        """Test that a new gaze aversion thread is created for each say call."""
        with patch("expressive_furhat.expressive_remote_api.FurhatRemoteAPI.say"):
            with patch(
                "expressive_furhat.expressive_remote_api.GazeAversionThread"
            ) as mock_gaze_class:
                mock_thread1 = Mock()
                mock_thread1.start = Mock()
                mock_thread1.join = Mock()

                mock_thread2 = Mock()
                mock_thread2.start = Mock()
                mock_thread2.join = Mock()

                mock_gaze_class.side_effect = [mock_thread1, mock_thread2]

                api.say(text="First", blocking=True)
                api.say(text="Second", blocking=True)

                # Two threads should be created
                assert mock_gaze_class.call_count == 2
                mock_thread1.start.assert_called_once()
                mock_thread2.start.assert_called_once()

    def test_conversation_format(self, api, mock_gesture_gen):
        """Test that conversation is formatted correctly for gesture generation."""
        with patch("expressive_furhat.expressive_remote_api.FurhatRemoteAPI.say"):
            with patch(
                "expressive_furhat.expressive_remote_api.GazeAversionThread"
            ) as mock_gaze:
                mock_thread = Mock()
                mock_thread.start = Mock()
                mock_thread.join = Mock()
                mock_gaze.return_value = mock_thread

                api.say(text="Hello!", blocking=True)
                api.react_to_text("Hi there!")

                # Check the formatted conversation passed to gesture gen
                call_arg = mock_gesture_gen.on_user_input.call_args[0][0]
                assert "Furhat: Hello!" in call_arg
                assert "user: Hi there!" in call_arg

    def test_react_to_text_without_prior_conversation(self, api, mock_gesture_gen):
        """Test reacting to text when conversation is empty."""
        assert len(api._conversation) == 0

        api.react_to_text("Hello!")

        assert len(api._conversation) == 1
        assert api._conversation[0]["role"] == "user"
        mock_gesture_gen.on_user_input.assert_called_once()

    def test_kwargs_passed_to_parent_say(self, api):
        """Test that arbitrary kwargs are passed to parent say method."""
        with patch(
            "expressive_furhat.expressive_remote_api.FurhatRemoteAPI.say"
        ) as mock_parent_say:
            with patch(
                "expressive_furhat.expressive_remote_api.GazeAversionThread"
            ) as mock_gaze:
                mock_thread = Mock()
                mock_thread.start = Mock()
                mock_thread.join = Mock()
                mock_gaze.return_value = mock_thread

                mock_parent_say.return_value = {"success": True}

                api.say(
                    text="Hello", blocking=True, custom_param="value", another_param=123
                )

                # Verify custom params were passed through
                call_kwargs = mock_parent_say.call_args[1]
                assert call_kwargs["text"] == "Hello"
                assert call_kwargs["custom_param"] == "value"
                assert call_kwargs["another_param"] == 123
