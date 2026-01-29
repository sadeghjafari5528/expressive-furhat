import threading
import time
from unittest.mock import Mock, patch

import pytest

from expressive_furhat.background_gestures import BackgroundGesturesThread
from expressive_furhat.expressive_remote_api import ExpressiveRemoteAPI
from expressive_furhat.gesture_generation import GestureGeneration
from tests.conftest import patch_furhat_init


class TestIntegration:
    """Integration tests for complete workflows."""

    @pytest.fixture
    def full_api_setup(self, mock_openai, mock_furhat):
        """Set up a full ExpressiveRemoteAPI with all components."""
        with patch_furhat_init():
            api = ExpressiveRemoteAPI(openai=mock_openai, with_gaze_aversion=True)
            # Mock the parent class methods
            api.gesture = Mock(return_value={"success": True})
            api.attend = Mock(return_value={"success": True})
            return api

    def test_gaze_aversion_with_gesture_generation(self, mock_openai, mock_furhat):
        """Test gaze aversion working alongside gesture generation."""
        with patch_furhat_init():
            with patch(
                "expressive_furhat.expressive_remote_api.FurhatRemoteAPI.say"
            ) as mock_parent_say:
                api = ExpressiveRemoteAPI(openai=mock_openai, with_gaze_aversion=True)
                api.gesture = mock_furhat.gesture
                api.attend = mock_furhat.attend

                mock_parent_say.return_value = {"success": True}

                # Mock gesture generation responses
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = "0"
                mock_openai.chat.completions.create = Mock(return_value=mock_response)

                # Say something
                api.say(text="Hello!", blocking=True)

                # Verify both gaze aversion and gesture generation were triggered
                assert mock_furhat.attend.call_count >= 1  # Eye contact restored

    def test_background_gestures_with_gesture_generation(
        self, mock_openai, mock_furhat
    ):
        """Test background gestures playing while new gestures are generated."""
        # Create a real BackgroundGesturesThread with short intervals
        bg_thread = BackgroundGesturesThread(mock_furhat, wait_seconds=0.1)

        # Add initial gestures
        sample_gesture = {
            "frames": [{"time": [0.0], "params": {"SMILE_OPEN": 0.5}}],
            "class": "furhatos.gestures.Gesture",
        }
        bg_thread.add_gesture_to_cache(0, sample_gesture)

        # Start the background thread
        bg_thread.start()

        # Let it play some gestures
        time.sleep(0.3)

        # Add more gestures while running (simulating gesture generation)
        for i in range(5):
            bg_thread.add_gesture_to_cache(i, sample_gesture)

        time.sleep(0.3)

        # Stop the thread
        bg_thread.join(timeout=1.0)

        # Verify gestures were played
        assert mock_furhat.gesture.call_count >= 2

        # Verify cache was populated
        assert len(bg_thread._gestures_cache) >= 5

    def test_mood_change_flushes_and_regenerates(self, mock_openai, mock_furhat):
        """Test that mood changes flush cache and trigger fast generation."""
        # Patch BackgroundGesturesThread to avoid real thread and double-start
        with patch(
            "expressive_furhat.gesture_generation.BackgroundGesturesThread"
        ) as mock_bg_class:
            mock_bg_thread = Mock()
            mock_bg_thread.add_gesture_to_cache = Mock()
            mock_bg_thread.flush_cache = Mock()
            mock_bg_thread.start = Mock()
            mock_bg_thread._gestures_cache = []
            mock_bg_class.return_value = mock_bg_thread

            # Create gesture generation
            gesture_gen = GestureGeneration(
                openai=mock_openai, furhat=mock_furhat, bg_thread=mock_bg_thread
            )

            # Mock mood change detection
            mock_mood_response = Mock()
            mock_mood_response.choices = [Mock()]
            mock_mood_response.choices[0].message.content = "1"  # Mood change

            mock_fast_response = Mock()
            mock_fast_response.choices = [Mock()]
            mock_fast_response.choices[
                0
            ].message.content = """Step 1: Sad
Step 2: {\"frames\": [{\"time\": [0.0, 1.0], \"params\": {\"EXPR_SAD\": 0.8}}], \"class\": \"furhatos.gestures.Gesture\"}"""

            mock_openai.chat.completions.create = Mock(
                side_effect=[mock_mood_response, mock_fast_response]
            )

            # Trigger mood check
            conversation = "user: I'm very sad now"
            gesture_gen._check_mood_change(conversation)

            # Wait for flush and fast generation
            time.sleep(0.1)

            # Cache should be flushed and new gesture added
            mock_bg_thread.flush_cache.assert_called()
            mock_bg_thread.add_gesture_to_cache.assert_called()

    def test_multiple_concurrent_say_calls(self, mock_openai, mock_furhat):
        """Test handling multiple concurrent say calls."""
        with patch_furhat_init():
            with patch(
                "expressive_furhat.expressive_remote_api.FurhatRemoteAPI.say"
            ) as mock_parent_say:
                api = ExpressiveRemoteAPI(openai=mock_openai, with_gaze_aversion=True)
                api.gesture = mock_furhat.gesture
                api.attend = mock_furhat.attend

                mock_parent_say.return_value = {"success": True}

                # Mock responses
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = "0"
                mock_openai.chat.completions.create = Mock(return_value=mock_response)

                # Spawn multiple say calls
                threads = []
                for i in range(3):
                    t = threading.Thread(
                        target=lambda idx: api.say(
                            text=f"Message {idx}", blocking=True
                        ),
                        args=(i,),
                    )
                    threads.append(t)
                    t.start()

                for t in threads:
                    t.join(timeout=2.0)

                # All should complete successfully
                assert len(api._conversation) == 3

    def test_weighted_gesture_sampling(self, mock_furhat):
        """Test that weighted sampling favors more recent gestures."""
        bg_thread = BackgroundGesturesThread(
            mock_furhat,
            weighting_factor=3.0,  # Strong preference for higher weights
            wait_seconds=0.05,
        )

        # Add gestures with different weights
        old_gesture = {
            "name": "old",
            "frames": [],
            "class": "furhatos.gestures.Gesture",
        }
        new_gesture = {
            "name": "new",
            "frames": [],
            "class": "furhatos.gestures.Gesture",
        }

        bg_thread.add_gesture_to_cache(0, old_gesture)  # weight = 3.0^0 = 1.0
        bg_thread.add_gesture_to_cache(10, new_gesture)  # weight = 3.0^10 = 59049

        bg_thread.start()
        time.sleep(0.5)
        bg_thread.join(timeout=1.0)

        # Count gestures played
        new_count = sum(
            1
            for call in mock_furhat.gesture.call_args_list
            if call[1]["body"]["name"] == "new"
        )

        # New gestures should be played much more frequently
        assert new_count >= mock_furhat.gesture.call_count * 0.8

    def test_pause_resume_during_conversation(self, mock_openai, mock_furhat):
        """Test pausing and resuming background gestures during conversation."""
        bg_thread = BackgroundGesturesThread(mock_furhat, wait_seconds=0.1)
        bg_thread.start()

        sample_gesture = {
            "frames": [{"time": [0.0], "params": {"SMILE_OPEN": 0.5}}],
            "class": "furhatos.gestures.Gesture",
        }
        bg_thread.add_gesture_to_cache(0, sample_gesture)

        # Let it play some gestures
        time.sleep(0.3)
        initial_count = mock_furhat.gesture.call_count

        # Pause
        bg_thread.pause()
        time.sleep(0.3)
        paused_count = mock_furhat.gesture.call_count

        # Should not play gestures while paused
        assert paused_count == initial_count

        # Resume
        bg_thread.resume()
        time.sleep(0.3)
        resumed_count = mock_furhat.gesture.call_count

        # Should play more gestures after resume
        assert resumed_count > paused_count

        bg_thread.join(timeout=1.0)

    def test_full_expressive_conversation(self, mock_openai, mock_furhat):
        """Test a complete expressive conversation with all features."""
        with patch_furhat_init():
            with patch(
                "expressive_furhat.expressive_remote_api.FurhatRemoteAPI.say"
            ) as mock_parent_say:
                # Create API
                api = ExpressiveRemoteAPI(openai=mock_openai, with_gaze_aversion=True)
                api.gesture = mock_furhat.gesture
                api.attend = mock_furhat.attend

                mock_parent_say.return_value = {"success": True}

                # Mock all OpenAI responses
                def create_response(content):
                    response = Mock()
                    response.choices = [Mock()]
                    response.choices[0].message.content = content
                    return response

                responses = [
                    create_response("0"),  # No mood change
                    create_response("Test behavior"),  # Reasoning
                    [],  # Streaming (empty for simplicity)
                ]
                mock_openai.chat.completions.create = Mock(side_effect=responses)

                # Simulate conversation
                api.say(text="Hello! How are you today?", blocking=True)
                api.react_to_text("I'm doing great, thanks!")
                api.say(text="That's wonderful to hear!", blocking=True)

                # Verify conversation tracking
                assert len(api._conversation) == 3

                # Verify gaze aversion occurred
                assert mock_furhat.attend.call_count >= 1
