import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FURHAT_IP = os.getenv("FURHAT_IP")

GESTURES = [
    'BigSmile',
    'Blink',
    'BrowFrown',
    'BrowRaise',
    'CloseEyes',
    'ExpressAnger',
    'ExpressDisgust',
    'ExpressFear',
    'ExpressSad',
    'GazeAway',
    'Nod',
    'Oh',
    'OpenEyes',
    'Roll',
    'Shake',
    'Smile',
    'Surprise',
    'Thoughtful',
    'Wink',
]