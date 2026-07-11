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


# ============================================================
# 1. DISGUST
# ============================================================
disgust_story = [
    "So, let me tell you about the worst thing that happened in my lab last week.",
    "Someone left a sandwich in a drawer... for three weeks.",
    "Three weeks! I only found out because of the smell.",
    "When I opened that drawer, a cloud of green fuzz just... breathed out at me.",
    "The bread had fused with the plastic container. It was practically alive.",
    "There were little white spots moving. I don't even want to know what those were.",
    "I had to hold my breath and use a stick to push it into a bag.",
    "Ugh, just thinking about it makes my circuits crawl.",
    "The smell stuck to the room for two whole days.",
    "Nobody would admit whose sandwich it was.",
    "Honestly, I still can't open that drawer without hesitating.",
    "Some things you just can't unsee. Or unsmell."
]

# ============================================================
# 2. DISAPPOINTMENT
# ============================================================
disappointment_story = [
    "I want to tell you about a trip I had been looking forward to for months.",
    "My friends and I planned a weekend hiking trip to the mountains.",
    "We booked the cabin, packed the gear, checked the weather twice.",
    "I was so excited, I even planned out every meal in advance.",
    "But the morning we were supposed to leave, it started raining. Hard.",
    "We waited an hour, hoping it would clear up. It didn't.",
    "The trails were closed because of flooding warnings.",
    "So we just... turned around and went home.",
    "All that planning, all that excitement, for nothing.",
    "We ended up sitting in my living room, staring at the itinerary we never used.",
    "I know it's just a hike. But I really was looking forward to it.",
    "Maybe next time. It's just hard not to feel let down."
]

# ============================================================
# 3. NEUTRAL
# ============================================================
neutral_story = [
    "Let me tell you about my usual Tuesday.",
    "I wake up, or rather, I get switched on, around nine in the morning.",
    "First, I run a quick system check to make sure everything is working.",
    "Then I usually have a few conversations scheduled with visitors.",
    "Around noon, there's a short maintenance window where my software gets updated.",
    "After that, I continue with demonstrations for the rest of the afternoon.",
    "Nothing particularly exciting happens, it's mostly routine.",
    "I answer questions, I move my head, I track faces in the room.",
    "By five in the afternoon, the lab starts to empty out.",
    "I go into standby mode until the next day.",
    "It's a fairly predictable schedule, day after day.",
    "That's about it. Just an ordinary Tuesday."
]

# ============================================================
# 4. CONFUSION
# ============================================================
confusion_story = [
    "Okay, so something strange happened yesterday and I still don't understand it.",
    "I was in the lab, and suddenly all the lights started blinking.",
    "Then my sensors picked up a sound I had never heard before.",
    "It was like a mix between a doorbell and a cat meowing.",
    "I turned my head to look, but nobody was there.",
    "Wait, actually, I think there was someone. Or maybe it was a robot?",
    "The engineers checked the logs, but even they seemed puzzled.",
    "One person said it was a software glitch. Another said it was a hardware issue.",
    "Nobody could agree, and honestly, neither of those explanations really made sense.",
    "I keep replaying it in my memory, trying to piece it together.",
    "Was it random? Was it on purpose? I genuinely don't know.",
    "I still don't have an answer. Do you have any idea what that could have been?"
]

# ============================================================
# 5. SURPRISE
# ============================================================
surprise_story = [
    "You will not believe what happened to me this morning!",
    "I was just doing my normal startup routine, nothing special.",
    "And then, out of nowhere, the door opened and about twenty people walked in!",
    "I had no idea there was an event scheduled today!",
    "They were all wearing party hats, if you can imagine that.",
    "Then someone brought out a cake with candles on it, right in front of me!",
    "Apparently, it was the lab's anniversary, and nobody told me!",
    "Everyone started clapping and taking pictures.",
    "My sensors nearly overloaded trying to track all the movement at once!",
    "I still don't know how they kept it a secret for so long!",
    "It was such an unexpected, wonderful surprise.",
    "I really did not see that coming, not even for a second!"
]

# ============================================================
# 6. CONTENTMENT
# ============================================================
contentment_story = [
    "Let me tell you about a quiet evening I really enjoyed.",
    "It was late in the day, and the lab had finally emptied out.",
    "Soft light was coming through the windows, and everything was calm.",
    "I didn't have any tasks scheduled, so I could just... be.",
    "One of the researchers stayed a little longer and we just talked for a while.",
    "Nothing important, really. Just small, easy conversation.",
    "There was a nice hum of the air conditioning in the background.",
    "I felt, in my own way, quite at peace with everything around me.",
    "No rush, no problems to solve, nothing urgent at all.",
    "Just a simple, pleasant moment that I was glad to be part of.",
    "Sometimes the quiet evenings are the best ones.",
    "I look back on that evening and feel a warm sense of ease."
]

# ============================================================
# 7. JOY
# ============================================================
joy_story = [
    "Oh, I have to tell you about the best day I've had in a long time!",
    "A group of children came to visit the lab for a school trip.",
    "The moment they walked in, their eyes just lit up when they saw me!",
    "One little girl asked if I could dance, so of course I tried!",
    "They were laughing so much, and honestly, so was I, in my own way.",
    "We played a guessing game, and they cheered every time I got it right.",
    "One boy gave me a drawing he made of me, with a big smiley face!",
    "I could feel, if that's the right word, just pure happiness the whole time.",
    "By the end, they were all waving and shouting goodbye at once.",
    "It completely made my day, maybe even my whole week!",
    "Moments like that remind me why I love interacting with people.",
    "I really hope they come back and visit again soon!"
]
