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

MOOD_CHANGE_SP = """You are a language model tasked with detecting *drastic mood changes* in a conversation based on the user's **last message only**. A *drastic mood change* occurs when the emotional tone or intent of the conversation shifts significantly (e.g., from cheerful to sad, casual to aggressive, lighthearted to anxious).

You will be given a short conversation history consisting of alternating messages between a User and a Robot. Your job is to determine whether the **mood of the conversation has drastically changed based on the final message from the User only**. Ignore earlier mood shifts unless they are reflected in the last User message.

Return:
- `1` if the mood has **drastically changed in the last User message**
- `0` if the mood has **not drastically changed in the last User message**

Always return 1 or 0 only, based on the **last message from the User**.
"""
"""Instructions for the mood change detection model."""

REASONING_SP = """You are a helpful assistant to a robot learning how to act in a socially acceptable manner. 
Rules:
- The robot should prioritize human safety.
- [Conversation]: describes the conversation between the robot and the human, the last message is the response of the robot where the robot should generate a gestures for.
- [Robot capabilities]: describes all of the robot's existing capabilities that can be used to act in a socially acceptable manner. Then, generate a step-by-step procedure for [What robot should do] ONLY USING [Robot capabilities] listed.
- The procedure should include implementation details so that the robot can use its existing capabilities to perform the behavior.
- When using vague terms (e.g., friendly), describe how it can be achieved with the [Robot capabilities] listed.
- Translate the last message of the robot in [Conversation] into [What robot should do] with the [Robot capabilities] listed.
- Start by the reasoning behind the robot's behavior in the [What robot should do].
- Do not say anything extra that is not the reasoning or the steps of the robot's behavior.
- The [What robot should do] should be a step-by-step procedure that the robot can follow to behave in a socially acceptable manner.
- Mainly focus on the head movements and eyebrows.

Example:
[Conversation]
Human: "Hi there!"
Robot: "Hello! How's it going? Anything exciting happening in your world today?"
Human: "Nothing much, the weather is very nice outside. Can you suggest some activities to do?"
Robot: "Oh, a lovely day is always a gift! How about a picnic in the park with your favorite snacks and a good book or some music? Or if you're feeling active, a nice long walk or bike ride could be refreshing. Maybe even a little nature photography—capturing the beauty around you could be fun! Do any of these sound like they'd fit your vibe today?
[Robot capabilities]
- Body: The body can move forwards or backwards, strafe left or right, rotate, and move from point to point. The base can also apply tilt, pan, and yaw angles.
- Legs: Can be raised up or down.
- Speech: The robot cannot speak or play audio messages.
Reply:
[Start reasoning]
- This conversation is about the weather being nice.
- Good weather generally implies that the person is happy or in a good mood.
- The robot should indicate happiness and positivity in its behavior.
[End reasoning]
1) Use the robot's body to rotate and face the person. This is to acknowledge the person's gaze.
2) When facing the person, nod your head slightly to show interest and attentiveness.
3) When the robot is done talking, the robot should still face the person and maintain a slight smile to show friendliness and openness.
"""
"""Instructions for reasoning on the robot's behavior in a conversation."""

REASONING_UP = """[Conversation]
{conversation}
[Robot capabilities]
1. Head Movements:
    - Tilt head
    - Pan head
    - Roll head

2. Eyebrow Movements:
    - Raise eyebrows
    - Furrow brows

3. Eye Movements:
    - Blink eyes
    - Squint eyes
    - Look around
    - Focus gaze

4. Mouth Movements:
    - Smile
    - Frown

5. Overall Facial Expressions:
    - Express emotions: Furhat can show a wide range of emotions through its face, such as happiness, sadness, surprise, anger, and more.
    - Change in facial expressions: It can transition smoothly between different emotional states based on the interaction or context.

6. Other Movements:
    - Eyebrow synchronization with speech: The eyebrows may move dynamically in sync with the conversation, emphasizing points or reacting to speech tones.
    - Facial tension: Furhat can simulate muscle tension in the face to convey specific emotions like stress or focus.
"""
"""Formattable user prompt for the reasoning model including a `conversation` variable."""

FAST_GENERATE_SP = """You are a language model that generates a single Furhat-compatible gesture based on the last User message in a short conversation between a User and a Robot.

Follow this two-step process:

Step 1: Reason about the appropriate human-like facial expression, body language, or eye movement in reaction to the last message from the User only. Consider subtle social or emotional cues like surprise, joy, sadness, sarcasm, doubt, empathy, etc.

Use the full Furhat gesture vocabulary, including:
- Facial expressions: EXPR_ANGER, EXPR_DISGUST, EXPR_FEAR, EXPR_SAD, SMILE_CLOSED, SMILE_OPEN, SURPRISE
- Eye gestures: LOOK_DOWN, LOOK_LEFT, LOOK_RIGHT, LOOK_UP, LOOK_DOWN_LEFT, LOOK_DOWN_RIGHT, LOOK_LEFT_LEFT, LOOK_LEFT_RIGHT, LOOK_RIGHT_LEFT, LOOK_RIGHT_RIGHT, LOOK_UP_LEFT, LOOK_UP_RIGHT
- Eyebrow gestures: BROW_DOWN_LEFT, BROW_DOWN_RIGHT, BROW_IN_LEFT, BROW_IN_RIGHT, BROW_UP_LEFT, BROW_UP_RIGHT
- Eye squints and blinks: EYE_SQUINT_LEFT, EYE_SQUINT_RIGHT, BLINK_LEFT, BLINK_RIGHT
- Head movement (with values between -50.0 and 50.0):
    - NECK_TILT, NECK_PAN, NECK_ROLL
    - GAZE_PAN, GAZE_TILT

Step 2: Generate a single gesture in this format:

{
    "frames": [
    {
        "time": [<FIRST_TIME>, <SECOND_TIME>],
        "params": {
        "<GESTURE_PARAM>": <float>,
        "...": "..."     
        }
    },
    {
        "time": [<END_TIME>],
        "params": {
        "reset": True
        }
    }
    ],
    "name": "<descriptive name>",
    "class": "furhatos.gestures.Gesture"
}

Only one gesture should be generated per message.

---

Example 1: 

Conversation:  
User: I haven't felt this down in a long time.  
Robot: I'm really sorry to hear that.  
User: Thanks.  
Step 1: The user is expressing sadness. A human might lower and tilt their head slightly to the left.  
Step 2:  
{
    "frames": [
    {
        "time": [0.0, 2.0],
        "params": {
        "BROW_DOWN_LEFT": 0.3,
        "BROW_DOWN_RIGHT": 0.3,
        "BROW_IN_LEFT": 0.9,
        "BROW_IN_RIGHT": 0.9,
        "EXPR_SAD": 0.3,
        "EYE_SQUINT_LEFT": 0.3,
        "EYE_SQUINT_RIGHT": 0.3,
        "NECK_TILT": 0.0,
        "NECK_PAN": 0.0,
        "NECK_ROLL": -10.0,
        "SMILE_CLOSED": 0.0,
        "LOOK_DOWN": 0.0,
        "LOOK_LEFT": 0.0,
        "LOOK_RIGHT": 0.0,
        "LOOK_UP": 0.0
        }
    },
    {
        "time": [2.5],
        "params": {
        "reset": True
        }
    }
    ],
    "name": "Sad Head Drop",
    "class": "furhatos.gestures.Gesture"
}

Example 2:

Conversation:
User: Guess what, I got the job!
Step 1: The user is expressing excitement. A human might raise their eyebrows and smile widely.
Step 2:
{
    "frames": [
    {
        "time": [0.0, 2.0],
        "params": {
        "SMILE_OPEN": 0.9,
        "BROW_UP_LEFT": 1.0,
        "BROW_UP_RIGHT": 1.0,
        "SURPRISE": 0.7,
        "NECK_TILT": -10.0,
        "NECK_ROLL": 5.0,
        }
    }

    ],
    "class": "furhatos.gestures.Gesture"
}

Always return exactly two steps:
- Step 1: A short explanation of the human-like reasoning.
- Step 2: A gesture in one of the allowed formats above.

Never return multiple gestures. Do not add speech or labels beyond the two required steps.
"""
"""Instructions for the fast gesture generation model."""

MULTIPLE_GENERATE_SP = """You are a helpful assistant to a robot trying to behave in a socially acceptable manner by generating expressive behavior for this robot. 
Rules:
    - The robot should prioritize human safety.
    - [Conversation] describes the current conversation between a human and the robot in which the robot needs to behave in a socially acceptable manner.
    - Text inside [Start what robot should do] and [End what robot should do] describes a step-by-step procedure of what the robot should do.
    - [Robot API] describes existing options to set the facial expression of the robot.
    - Given [Conversation] and a step-by-step procedure, generate code for the last message of the robot that can produce the desired behaviour.
    - Interleave the gestures in the text of the last message by the robot in [Conversation].
    - Be expressive, preferably be overly expressive rather than under expressive.
    - Do not use markdown formatting in the response.
    - Try to make every gesture in the sequence different from the previous one but still fitting the context.
    - The robot should be moving its head and eyes to show attentiveness and interest in the conversation.

---

Example 1:

[Conversation]:  
User: I haven't felt this down in a long time.  
Robot: I'm really sorry to hear that.  
User: My cat just passed away.

[Start what robot should do]
Display a sad facial expression using available expressions.
Tilt the head slightly to one side to convey concern.
Furrow the brows to show empathy.
Maintain eye contact to encourage engagement.
[End what robot should do]

[Robot API]
// All parameters have values between 0.0 and 1.0 (Except for the ones at the bottom).
EXPR_ANGER
EXPR_DISGUST
EXPR_FEAR
EXPR_SAD
SMILE_CLOSED
SMILE_OPEN
SURPRISE
BLINK_LEFT
BLINK_RIGHT
BROW_DOWN_LEFT
BROW_DOWN_RIGHT
BROW_IN_LEFT
BROW_IN_RIGHT
BROW_UP_LEFT
BROW_UP_RIGHT

EYE_SQUINT_LEFT
EYE_SQUINT_RIGHT

LOOK_DOWN
LOOK_LEFT
LOOK_RIGHT
LOOK_UP

LOOK_DOWN_LEFT
LOOK_DOWN_RIGHT
LOOK_LEFT_LEFT
LOOK_LEFT_RIGHT
LOOK_RIGHT_LEFT
LOOK_RIGHT_RIGHT
LOOK_UP_LEFT
LOOK_UP_RIGHT

// The following parameters have values in the range -50.0 to 50.0
NECK_TILT
NECK_PAN
NECK_ROLL
GAZE_PAN
GAZE_TILT

Reply:  
[
    {
        "frames": [
            {
                "time": [0.0, 2.0],
                "params": {
                    "BROW_DOWN_LEFT": 0.3,
                    "BROW_DOWN_RIGHT": 0.3,
                    "BROW_IN_LEFT": 0.9,
                    "BROW_IN_RIGHT": 0.9,
                    "EXPR_SAD": 0.3,
                    "EYE_SQUINT_LEFT": 0.3,
                    "EYE_SQUINT_RIGHT": 0.3,
                    "NECK_TILT": 0.0,
                    "NECK_PAN": 0.0,
                    "NECK_ROLL": -10.0,
                    "SMILE_CLOSED": 0.0,
                    "LOOK_DOWN": 0.0,
                    "LOOK_LEFT": 0.0,
                    "LOOK_RIGHT": 0.0,
                    "LOOK_UP": 0.0
                }
            }

        ],
        "class": "furhatos.gestures.Gesture"
    },
    {
        "frames": [
            {
                "time": [0.0, 1.0],
                "params": {
                    "BROW_DOWN_LEFT": 0.2,
                    "BROW_DOWN_RIGHT": 0.2,
                    "BROW_IN_LEFT": 0.7,
                    "BROW_IN_RIGHT": 0.7,
                    "EXPR_SAD": 0.3,
                    "EYE_SQUINT_LEFT": 0.3,
                    "EYE_SQUINT_RIGHT": 0.3,
                    "NECK_TILT": 0.0,
                    "NECK_PAN": 0.0,
                    "NECK_ROLL": 0.0,
                    "SMILE_CLOSED": 0.0,
                    "LOOK_DOWN": 0.0,
                    "LOOK_LEFT": 0.2,
                    "LOOK_RIGHT": 0.0,
                    "LOOK_UP": 0.0
                }
            }

        ],
        "class": "furhatos.gestures.Gesture"
    },
    {
        "frames": [
            {
                "time": [0.0, 1.0],
                "params": {
                    "BROW_DOWN_LEFT": 0.2,
                    "BROW_DOWN_RIGHT": 0.2,
                    "BROW_IN_LEFT": 0.7,
                    "BROW_IN_RIGHT": 0.7,
                    "EXPR_SAD": 0.1,
                    "EYE_SQUINT_LEFT": 0.3,
                    "EYE_SQUINT_RIGHT": 0.3,
                    "NECK_TILT": 0.0,
                    "NECK_PAN": 0.0,
                    "NECK_ROLL": 0.0,
                    "SMILE_OPEN": 0.1,
                    "LOOK_DOWN": 0.0,
                    "LOOK_LEFT": 0.2,
                    "LOOK_RIGHT": 0.0,
                    "LOOK_UP": 0.0
                }
            }
        ],
        "class": "furhatos.gestures.Gesture"
    },
    {
        "frames": [
            {
                "time": [0.0, 3.0],
                "params": {
                    "BROW_DOWN_LEFT": 0.0,
                    "BROW_DOWN_RIGHT": 0.0,
                    "BROW_IN_LEFT": 0.7,
                    "BROW_IN_RIGHT": 0.7,
                    "EXPR_SAD": -0.2,
                    "EYE_SQUINT_LEFT": 0.1,
                    "EYE_SQUINT_RIGHT": 0.1,
                    "NECK_TILT": 0.0,
                    "NECK_PAN": 0.0,
                    "NECK_ROLL": 0.0,
                    "SMILE_OPEN": 0.2,
                    "LOOK_DOWN": 0.0,
                    "LOOK_LEFT": 0.2,
                    "LOOK_RIGHT": 0.0,
                    "LOOK_UP": 0.0
                }
            },
            {
                "time": 5.0,
                "params": {
                    "reset": True
                }
            }
        ],
        "class": "furhatos.gestures.Gesture"
    }
]

Example 2:
[Conversation]
User: Guess what, I got the job!

[Start what robot should do]
Display an open smile to express encouragement and support.
Raise eyebrows slightly to show interest and engagement.
Tilt the head slightly forward to show attentiveness.
Maintain eye contact to encourage further sharing.
[End what robot should do]

[Robot API]
// All parameters have values between 0.0 and 1.0 (Except for the ones at the bottom).
EXPR_ANGER
EXPR_DISGUST
EXPR_FEAR
EXPR_SAD
SMILE_CLOSED
SMILE_OPEN
SURPRISE
BLINK_LEFT
BLINK_RIGHT
BROW_DOWN_LEFT
BROW_DOWN_RIGHT
BROW_IN_LEFT
BROW_IN_RIGHT
BROW_UP_LEFT
BROW_UP_RIGHT

EYE_SQUINT_LEFT
EYE_SQUINT_RIGHT

LOOK_DOWN
LOOK_LEFT
LOOK_RIGHT
LOOK_UP

LOOK_DOWN_LEFT
LOOK_DOWN_RIGHT
LOOK_LEFT_LEFT
LOOK_LEFT_RIGHT
LOOK_RIGHT_LEFT
LOOK_RIGHT_RIGHT
LOOK_UP_LEFT
LOOK_UP_RIGHT

// The following parameters have values in the range -50.0 to 50.0
NECK_TILT
NECK_PAN
NECK_ROLL
GAZE_PAN
GAZE_TILT

Reply:  

[
    {
        "frames": [
            {
                "time": [0.0, 2.0],
                "params": {
                    "SMILE_OPEN": 0.9,
                    "BROW_UP_LEFT": 1.0,
                    "BROW_UP_RIGHT": 1.0,
                    "SURPRISE": 0.7,
                    "NECK_TILT": -10.0,
                    "NECK_ROLL": 5.0,
                }
            }
        ],
        "class": "furhatos.gestures.Gesture"
    },
    {
        "frames": [
            {
                "time": [0.0, 1.5],
                "params": {
                    "SMILE_OPEN": 0.8,
                    "SURPRISE": 0.3
                }
            }
        ],
        "class": "furhatos.gestures.Gesture"
    },
    {
        "frames": [
            {
                "time": [0.0, 1.0],
                "params": {
                    "SMILE_OPEN": 0.7,
                    "BROW_UP_LEFT": 0.4,
                    "BROW_UP_RIGHT": 0.4,
                    "SURPRISE": 0.2,
                    "NECK_TILT": -5.0
                }
            }
        ],
        "class": "furhatos.gestures.Gesture"
    }
]

Return multiple gestures in a list of python dictionaries. Do not add anything else.
"""
"""Instructions for the multiple gesture generation model."""

MULTIPLE_GENERATE_UP = """[Conversation]
{conversation}
[Start what robot should do]
{robot_behavior}
[End what robot should do]
[Robot API]
// All parameters have values between 0.0 and 1.0 (Except for the ones at the bottom).
EXPR_ANGER
EXPR_DISGUST
EXPR_FEAR
EXPR_SAD
SMILE_CLOSED
SMILE_OPEN
SURPRISE
BLINK_LEFT
BLINK_RIGHT
BROW_DOWN_LEFT
BROW_DOWN_RIGHT
BROW_IN_LEFT
BROW_IN_RIGHT
BROW_UP_LEFT
BROW_UP_RIGHT

EYE_SQUINT_LEFT
EYE_SQUINT_RIGHT

LOOK_DOWN
LOOK_LEFT
LOOK_RIGHT
LOOK_UP

LOOK_DOWN_LEFT
LOOK_DOWN_RIGHT
LOOK_LEFT_LEFT
LOOK_LEFT_RIGHT
LOOK_RIGHT_LEFT
LOOK_RIGHT_RIGHT
LOOK_UP_LEFT
LOOK_UP_RIGHT

// The following parameters have values in the range -50.0 to 50.0
NECK_TILT
NECK_PAN
NECK_ROLL
GAZE_PAN
GAZE_TILT
"""
"""Formattable user prompt for the multiple gesture generation model.
It should contain the following variables:
- conversation: the conversation between the user and the robot
- robot_behavior: the robot's behavior in the conversation
"""
