# Expressive Furhat

A Python library that enhances the Furhat robot's expressiveness through AI-generated gestures and natural gaze aversion.
This implementation is a drop-in replacement for the standard `furhat-remote-api` library.

## Paper Reference

> **Expressive Furhat: Generating Real-Time Facial Expressions for Human-Robot Dialogue with LLMs**  
> Giulio Antonio Abbo, Ruben Janssens, Seppe Van de Vreken, and Tony Belpaeme  
> _To be presented at the 21st ACM/IEEE International Conference on Human-Robot Interaction (HRI 2026)_
>
> _Paper reference will be added upon publication._

## Features

- **Gaze Aversion**: The robot looks away from users during speech, mimicking natural human conversational behaviour
- **AI-Generated Gestures**: Multi-stage LLM pipeline that detects mood changes and generates appropriate facial expressions and head movements
- **Easy Integration**: Drop-in replacement for `FurhatRemoteAPI` with minimal code changes
- **Thread-Safe**: All expressive features run efficiently in background threads
- **Configurable**: Customizable timing, gesture intensity, and LLM models (supports OpenAI and Ollama)

## Requirements

- Python 3.10 or higher (tested on Python 3.13.7)
- Furhat robot or Furhat SDK simulation (free version 2.8.4 or higher from: https://www.furhatrobotics.com/requestsdk)
- OpenAI API key (or Ollama for local LLM execution)

## Installation

### 1. Install Dependencies

Clone this repository and install the required Python packages, or install via PyPI (`pip install expressive-furhat`).

```bash
git clone <repository-url>
cd seppeGenEM
pip install -r requirements.txt
```

### 2. Set Up Furhat

- Start your Furhat robot or launch the Furhat SDK simulation
- Enable Remote APIs in Furhat settings (Instructions: https://docs.furhat.io/remote-api/)
- If using a physical robot, note its IP address (default is `localhost` for simulation)

### 3. Configure API Keys

Set your OpenAI API key in the code or as an environment variable, or configure Ollama for local LLM usage.

## Usage

The included `main.py` provides an example of how to use the `ExpressiveRemoteAPI`, which wraps all the functionalities in a single class. Here is a minimal example.

```python
from furhat_remote_api import FurhatRemoteAPI
from openai import OpenAI
from expressive_furhat.expressive_remote_api import ExpressiveRemoteAPI

client = OpenAI(api_key=api_key)
furhat = ExpressiveRemoteAPI(furhat_ip, openai=client)

message = "Hi there! What would you like to talk about?"
conversation = [
    {"role": "system", "content": "You are Furhat, a kind and friendly social robot"},
    {"role": "assistant", "content": message},
]
furhat.say(text=message, blocking=True)

user_message = furhat.listen().message
conversation.append({"role": "user", "content": user_message})
furhat.react_to_text(user_message)
```

The `ExpressiveRemoteAPI` extends `FurhatRemoteAPI` with:

- Automatic gaze aversion when using `say()` method
- Gesture generation triggered by conversation context
- Conversation history tracking for contextual expressiveness

Additionally, `BackgroundGesturesThread`, `GazeAversionThread`, and `GestureGeneration` can be used independently for more granular control.

## Documentation

For detailed documentation see the inline documentation in the `expressive_furhat/` folder.

## Project Structure

```
├── expressive_furhat/          # Main library code
│   ├── expressive_remote_api.py   # Main API wrapper
│   ├── gaze_aversion.py           # Gaze control thread
│   ├── gesture_generation.py      # LLM-based gesture pipeline
│   ├── background_gestures.py     # Gesture caching and playback
│   └── prompts.py                 # LLM system prompts
├── main.py                     # Tutorial and demo
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests. Make sure to follow the existing code style and keep the documentation and tests up to date.

## Testing

Run the test suite using `pytest`:

```bash
pip install pytest
pytest tests/
```
