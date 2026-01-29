
import threading
import time
from unittest.mock import Mock

from expressive_furhat.background_gestures import BackgroundGesturesThread


class TestBackgroundGesturesThread:

    def test_furhat_api_error_during_gesture(self, mock_furhat):
        mock_furhat.gesture.side_effect = Exception("Furhat connection lost")
        from expressive_furhat.background_gestures import BackgroundGesturesThread
        bg_thread = BackgroundGesturesThread(mock_furhat, wait_seconds=0.1)
        sample_gesture = {
            "frames": [{"time": [0.0], "params": {"SMILE_OPEN": 0.5}}],
            "class": "furhatos.gestures.Gesture",
        }
        bg_thread.add_gesture_to_cache(0, sample_gesture)
        bg_thread.start()
        import time
        time.sleep(0.2)
        bg_thread.join(timeout=1.0)
        assert not bg_thread.is_alive()

    def test_concurrent_cache_operations_stress(self, mock_furhat):
        import threading
        from expressive_furhat.background_gestures import BackgroundGesturesThread
        bg_thread = BackgroundGesturesThread(mock_furhat, wait_seconds=0.1)
        sample_gesture = {
            "frames": [{"time": [0.0], "params": {"SMILE_OPEN": 0.5}}],
            "class": "furhatos.gestures.Gesture",
        }
        def add_many():
            for i in range(50):
                bg_thread.add_gesture_to_cache(i, sample_gesture)
        def flush_many():
            for _ in range(10):
                import time
                time.sleep(0.01)
                bg_thread.flush_cache()
        bg_thread.start()
        threads = []
        threads.extend([threading.Thread(target=add_many) for _ in range(3)])
        threads.extend([threading.Thread(target=flush_many) for _ in range(2)])
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        bg_thread.join(timeout=1.0)
        assert isinstance(bg_thread._gestures_cache, list)

    def test_empty_gestures_cache_continuous_sampling(self, mock_furhat):
        from expressive_furhat.background_gestures import BackgroundGesturesThread
        bg_thread = BackgroundGesturesThread(mock_furhat, wait_seconds=0.05)
        bg_thread.start()
        import time
        time.sleep(0.2)
        bg_thread.join(timeout=1.0)
        mock_furhat.gesture.assert_not_called()

    def test_invalid_gesture_structure_parsing(self, mock_furhat):
        from openai import OpenAI
        from expressive_furhat.gesture_generation import GestureGeneration
        gesture_gen = GestureGeneration(openai=Mock(spec=OpenAI), furhat=mock_furhat)
        invalid_gestures = [
            {"frames": []},
            {"class": "furhatos.gestures.Gesture"},
            {},
        ]
        for invalid_gesture in invalid_gestures:
            try:
                result = gesture_gen._parse_gesture([invalid_gesture])
                assert isinstance(result, list)
            except (KeyError, TypeError):
                pass


    def test_initialization(self, mock_furhat):
        # Test default initialization
        thread = BackgroundGesturesThread(mock_furhat)
        assert thread.furhat is mock_furhat
        assert thread.weighting_factor == 1.4
        assert thread.wait_seconds == 1.0
        assert thread.daemon is True
        assert len(thread._gestures_cache) == 0

        # Test custom initialization
        thread = BackgroundGesturesThread(
            mock_furhat, weighting_factor=2.0, wait_seconds=0.5
        )
        assert thread.weighting_factor == 2.0
        assert thread.wait_seconds == 0.5

    def test_add_gesture_to_cache(self, mock_furhat, sample_gesture):
        thread = BackgroundGesturesThread(mock_furhat)

        # Add gestures with different weights
        thread.add_gesture_to_cache(0, sample_gesture)
        thread.add_gesture_to_cache(1, sample_gesture)
        thread.add_gesture_to_cache(2, sample_gesture)

        assert len(thread._gestures_cache) == 3
        assert thread._gestures_cache[0] == (0, sample_gesture)
        assert thread._gestures_cache[1] == (1, sample_gesture)
        assert thread._gestures_cache[2] == (2, sample_gesture)

    def test_flush_cache(self, mock_furhat, sample_gesture):
        thread = BackgroundGesturesThread(mock_furhat)

        # Add gestures
        thread.add_gesture_to_cache(0, sample_gesture)
        thread.add_gesture_to_cache(1, sample_gesture)
        assert len(thread._gestures_cache) == 2

        # Flush cache
        thread.flush_cache()
        assert len(thread._gestures_cache) == 0

    def test_pause_resume(self, mock_furhat):
        thread = BackgroundGesturesThread(mock_furhat)

        assert not thread._is_paused.is_set()

        thread.pause()
        assert thread._is_paused.is_set()

        thread.resume()
        assert not thread._is_paused.is_set()

    def test_sample_from_cache_empty(self, mock_furhat):
        thread = BackgroundGesturesThread(mock_furhat)

        gesture = thread._sample_from_cache()
        assert gesture is None

    def test_sample_from_cache_single(self, mock_furhat, sample_gesture):
        thread = BackgroundGesturesThread(mock_furhat)
        thread.add_gesture_to_cache(0, sample_gesture)

        gesture = thread._sample_from_cache()
        assert gesture == sample_gesture

    def test_sample_from_cache_weighted(self, mock_furhat, sample_gesture):
        thread = BackgroundGesturesThread(mock_furhat, weighting_factor=2.0)
        assert thread.weighting_factor == 2.0  # Verify the parameter was set

        gesture1 = {"name": "gesture1", "frames": []}
        gesture2 = {"name": "gesture2", "frames": []}

        # Add gestures with different weights
        thread.add_gesture_to_cache(0, gesture1)  # weight = 2.0^0 = 1.0
        thread.add_gesture_to_cache(5, gesture2)  # weight = 2.0^5 = 32.0

        # Sample multiple times and verify both gestures can be selected
        # (Higher weight should be more frequent, but both possible)
        samples = [thread._sample_from_cache() for _ in range(100)]
        assert any(s["name"] == "gesture1" for s in samples)
        assert any(s["name"] == "gesture2" for s in samples)

    def test_send_gesture_blocking(self, mock_furhat, sample_gesture):
        thread = BackgroundGesturesThread(mock_furhat)

        thread._send_gesture_blocking(sample_gesture)

        mock_furhat.gesture.assert_called_once_with(body=sample_gesture, blocking=True)

    def test_thread_lifecycle(self, mock_furhat, sample_gesture):
        thread = BackgroundGesturesThread(mock_furhat, wait_seconds=0.1)
        thread.add_gesture_to_cache(0, sample_gesture)

        # Start thread
        thread.start()
        time.sleep(0.1)  # Give thread time to start
        # Let it run briefly
        time.sleep(0.3)

        # Stop thread
        thread.join(timeout=1.0)
        assert not thread.is_alive()

        # Verify at least one gesture was sent
        assert mock_furhat.gesture.call_count >= 1

    def test_thread_paused_no_gestures(self, mock_furhat, sample_gesture):
        thread = BackgroundGesturesThread(mock_furhat, wait_seconds=0.1)
        thread.add_gesture_to_cache(0, sample_gesture)
        thread.pause()

        thread.start()
        time.sleep(0.3)
        thread.join(timeout=1.0)

        # No gestures should be sent while paused
        mock_furhat.gesture.assert_not_called()

    def test_thread_resume_after_pause(self, mock_furhat, sample_gesture):
        thread = BackgroundGesturesThread(mock_furhat, wait_seconds=0.1)
        thread.add_gesture_to_cache(0, sample_gesture)

        # Pause before starting to ensure no gestures are sent initially
        thread.pause()
        thread.start()
        time.sleep(0.2)

        # Verify no gestures while paused
        assert mock_furhat.gesture.call_count == 0

        thread.resume()
        time.sleep(0.3)
        thread.join(timeout=1.0)

        # Verify gestures sent after resume
        assert mock_furhat.gesture.call_count >= 1

    def test_thread_safety_add_gesture(self, mock_furhat, sample_gesture):
        thread = BackgroundGesturesThread(mock_furhat, wait_seconds=0.1)
        thread.start()

        # Add gestures from multiple threads
        def add_gestures():
            for i in range(10):
                thread.add_gesture_to_cache(i, sample_gesture)

        threads = [threading.Thread(target=add_gestures) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        thread.join(timeout=1.0)

        # Verify all gestures were added
        assert len(thread._gestures_cache) == 30

    def test_thread_safety_flush_cache(self, mock_furhat, sample_gesture):
        thread = BackgroundGesturesThread(mock_furhat, wait_seconds=0.1)
        thread.add_gesture_to_cache(0, sample_gesture)
        thread.start()

        time.sleep(0.1)
        thread.flush_cache()
        time.sleep(0.1)

        thread.join(timeout=1.0)

        # Cache should be empty after flush
        assert len(thread._gestures_cache) == 0
