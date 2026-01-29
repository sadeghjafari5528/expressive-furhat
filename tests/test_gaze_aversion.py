import time

from expressive_furhat.gaze_aversion import GazeAversionThread


class TestGazeAversionThread:
    """Test suite for GazeAversionThread."""

    def test_initialization_defaults(self, mock_furhat):
        """Test thread initialization with default parameters."""
        thread = GazeAversionThread(mock_furhat)

        assert thread.furhat is mock_furhat
        assert thread.engaged_duration_range == [2.0, 5.0]
        assert thread.eversion_duration_range == [0.5, 4.0]
        assert thread.pan_range == [-20, 20]
        assert thread.tilt_range == [-30, 30]
        assert thread.eye_movement_speed == 0.2
        assert thread.daemon is True

    def test_initialization_custom(self, mock_furhat):
        """Test thread initialization with custom parameters."""
        thread = GazeAversionThread(
            mock_furhat,
            engaged_duration_range=[1.0, 2.0],
            eversion_duration_range=[0.2, 1.0],
            pan_range=[-10, 10],
            tilt_range=[-15, 15],
            eye_movement_speed=0.5,
        )

        assert thread.engaged_duration_range == [1.0, 2.0]
        assert thread.eversion_duration_range == [0.2, 1.0]
        assert thread.pan_range == [-10, 10]
        assert thread.tilt_range == [-15, 15]
        assert thread.eye_movement_speed == 0.5

    def test_gaze_aversion_resumed(self, mock_furhat):
        """Test gaze aversion gesture generation."""
        thread = GazeAversionThread(
            mock_furhat,
            pan_range=[-20, 20],
            tilt_range=[-30, 30],
            eye_movement_speed=0.2,
        )

        thread.gaze_aversion_resumed(duration=1.0)

        # Verify gesture was called
        assert mock_furhat.gesture.called
        call_kwargs = mock_furhat.gesture.call_args[1]

        # Verify blocking=False
        assert call_kwargs["blocking"] is False

        # Verify gesture structure
        gesture = call_kwargs["body"]
        assert "name" in gesture
        assert gesture["name"] == "GazeAversion"
        assert "class" in gesture
        assert "frames" in gesture
        assert len(gesture["frames"]) == 2

        # Verify first frame has GAZE_PAN and GAZE_TILT
        first_frame = gesture["frames"][0]
        assert "GAZE_PAN" in first_frame["params"]
        assert "GAZE_TILT" in first_frame["params"]

        # Verify values are within ranges
        assert -20 <= first_frame["params"]["GAZE_PAN"] <= 20
        assert -30 <= first_frame["params"]["GAZE_TILT"] <= 30

        # Verify second frame has reset
        second_frame = gesture["frames"][1]
        assert second_frame["params"]["reset"] is True

    def test_gaze_aversion_paused(self, mock_furhat):
        """Test eye contact restoration."""
        thread = GazeAversionThread(mock_furhat)

        thread.gaze_aversion_paused()

        # Verify attend was called for closest user
        mock_furhat.attend.assert_called_once_with(user="CLOSEST")

    def test_thread_lifecycle(self, mock_furhat):
        """Test thread start and stop."""
        thread = GazeAversionThread(
            mock_furhat,
            engaged_duration_range=[0.1, 0.2],
            eversion_duration_range=[0.1, 0.2],
        )

        thread.start()
        time.sleep(0.1)  # Give thread time to start

        # Stop thread
        thread.join(timeout=1.0)
        assert not thread.is_alive()

    def test_thread_alternates_gaze(self, mock_furhat):
        """Test that thread alternates between engaged and averted gaze."""
        thread = GazeAversionThread(
            mock_furhat,
            engaged_duration_range=[0.1, 0.15],
            eversion_duration_range=[0.1, 0.15],
        )

        thread.start()
        time.sleep(0.6)  # Should allow at least 2 cycles
        thread.join(timeout=1.0)

        # Verify both gesture (aversion) and attend (engagement) were called
        assert mock_furhat.gesture.call_count >= 1
        assert mock_furhat.attend.call_count >= 1

    def test_thread_stops_immediately_on_join(self, mock_furhat):
        """Test that thread stops quickly when join is called."""
        thread = GazeAversionThread(
            mock_furhat,
            engaged_duration_range=[10.0, 20.0],  # Long durations
            eversion_duration_range=[10.0, 20.0],
        )

        thread.start()
        time.sleep(0.1)

        start_time = time.time()
        thread.join(timeout=2.0)
        stop_time = time.time()

        # Should stop in less than 1 second despite long durations
        assert stop_time - start_time < 1.0
        assert not thread.is_alive()

    def test_thread_restores_eye_contact_on_stop(self, mock_furhat):
        """Test that thread restores eye contact when stopped."""
        thread = GazeAversionThread(
            mock_furhat,
            engaged_duration_range=[0.1, 0.2],
            eversion_duration_range=[0.1, 0.2],
        )

        thread.start()
        time.sleep(0.3)

        # Reset call count before stopping
        mock_furhat.attend.reset_mock()

        thread.join(timeout=1.0)

        # Verify attend was called to restore eye contact
        assert mock_furhat.attend.called
        assert mock_furhat.attend.call_args[1]["user"] == "CLOSEST"

    def test_custom_gaze_aversion_methods(self, mock_furhat):
        """Test that custom gaze aversion methods can be overridden."""

        class CustomGazeAversionThread(GazeAversionThread):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.resumed_calls = []
                self.paused_calls = []

            def gaze_aversion_resumed(self, duration):
                self.resumed_calls.append(duration)

            def gaze_aversion_paused(self, duration=None):
                self.paused_calls.append(duration)

        thread = CustomGazeAversionThread(
            mock_furhat,
            engaged_duration_range=[0.1, 0.15],
            eversion_duration_range=[0.1, 0.15],
        )

        thread.start()
        time.sleep(0.5)
        thread.join(timeout=1.0)

        # Verify custom methods were called
        assert len(thread.resumed_calls) >= 1
        assert len(thread.paused_calls) >= 2  # Once during run, once on stop

    def test_random_gaze_directions(self, mock_furhat):
        """Test that gaze aversion uses random directions."""
        thread = GazeAversionThread(
            mock_furhat, pan_range=[-20, 20], tilt_range=[-30, 30]
        )

        # Generate multiple gaze aversions
        directions = []
        for _ in range(10):
            mock_furhat.gesture.reset_mock()
            thread.gaze_aversion_resumed(duration=1.0)

            call_kwargs = mock_furhat.gesture.call_args[1]
            gesture = call_kwargs["body"]
            first_frame = gesture["frames"][0]

            directions.append(
                (first_frame["params"]["GAZE_PAN"], first_frame["params"]["GAZE_TILT"])
            )

        # Verify not all directions are the same (randomness)
        unique_directions = set(directions)
        assert len(unique_directions) > 1

    def test_timing_parameters(self, mock_furhat):
        """Test that timing parameters are correctly used in gestures."""
        eye_speed = 0.3
        duration = 2.5

        thread = GazeAversionThread(mock_furhat, eye_movement_speed=eye_speed)

        thread.gaze_aversion_resumed(duration=duration)

        call_kwargs = mock_furhat.gesture.call_args[1]
        gesture = call_kwargs["body"]

        # Verify first frame timing
        first_frame = gesture["frames"][0]
        assert first_frame["time"] == [0.0, eye_speed]

        # Verify second frame timing
        second_frame = gesture["frames"][1]
        expected_start = eye_speed + duration
        expected_end = 2 * eye_speed + duration
        assert second_frame["time"] == [expected_start, expected_end]
