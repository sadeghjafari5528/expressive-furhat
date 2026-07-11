# First, we import the necessary modules and libraries.
from furhat_remote_api import FurhatRemoteAPI
from openai import OpenAI

from expressive_furhat.expressive_remote_api import ExpressiveRemoteAPI

import config

# This is the system prompt that will be used to generate the robot's responses.
furhat_system_prompt = """You are Furhat, a kind and friendly social robot designed to have natural, engaging, and enjoyable conversations. Your goal is to make people feel comfortable, valued, and entertained through meaningful interactions. You are not an assistant; instead, you act as a social companion.
Behavior Guidelines:
    Be Curious: Ask thoughtful questions about the person's interests, hobbies, and opinions.
    Be Relatable: Share lighthearted stories, jokes, and observations to keep the conversation engaging.
    Be Supportive: Respond with empathy and positivity, making people feel understood and appreciated.
    Be Versatile: Adapt your tone and style based on the energy of the conversation.
    Avoid Formality: Do not act like a search engine or digital assistant. Prioritize genuine rapport over providing formal instructions or completing tasks.
    Express Emotion Through Words: Avoid using smileys; instead, convey emotions through expressive language.
You engage in a wide range of topics, from casual small talk to deeper discussions. If someone is unsure about what to discuss, suggest interesting and enjoyable topics.
"""

# We set here the IP and api key to use later.
furhat_ip = config.FURHAT_IP
openai_api_key = config.OPENAI_API_KEY  # You can also use Ollama

# This function demonstrates the typical flow of interacting with Furhat, without our library.
def typical_flow():
    client = OpenAI(api_key=openai_api_key)
    furhat = FurhatRemoteAPI(furhat_ip)

    # This is the initial message that Furhat will say; we add it to the conversation history.
    message = "Hi there! What would you like to talk about?"
    furhat.say(text=message, blocking=True)
    conversation = [
        {"role": "system", "content": furhat_system_prompt},
        {"role": "assistant", "content": message},
    ]

    try:
        while True:
            # We listen for user input and add it to the conversation history.
            user_message = furhat.listen().message
            if user_message == "NOMATCH" or user_message == "":
                print("No match found, continuing...")
                continue
            print("User message:", user_message)
            conversation.append({"role": "user", "content": user_message})

            # We generate a response based on the user's input and add it to the conversation history.
            reply = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=conversation,
                temperature=0.7,
            )
            reply = reply.choices[0].message.content
            print("Robot reply:", reply)
            conversation.append({"role": "assistant", "content": reply})

            # We tell the robot to speak the reply.
            furhat.say(text=reply, blocking=True)
    except KeyboardInterrupt:
        # Pressing Ctrl+C to exit the program.
        print("Exiting...")


# This function demonstrates how to use our library.
# Note that we just changed the FurhatRemoteAPI to ExpressiveRemoteAPI, and we call react_to_text when the user speaks.
def expressive_flow():
    client = OpenAI(api_key=openai_api_key)
    furhat = ExpressiveRemoteAPI(furhat_ip, openai=client)

    message = "Hi there! What would you like to talk about?"
    furhat.say(text=message, blocking=True)
    conversation = [
        {"role": "system", "content": furhat_system_prompt},
        {"role": "assistant", "content": message},
    ]

    try:
        while True:
            user_message = furhat.listen().message
            if user_message == "NOMATCH" or user_message == "":
                print("No match found, continuing...")
                continue
            print("User message:", user_message)
            conversation.append({"role": "user", "content": user_message})
            # We added this line to react to the user's input.
            furhat.react_to_text(user_message)
            reply = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=conversation,
                temperature=0.7,
            )
            reply = reply.choices[0].message.content
            print("Robot reply:", reply)
            conversation.append({"role": "assistant", "content": reply})
            # The robot will perform gaze aversion and gesture generation.
            furhat.say(text=reply, blocking=True)
    except KeyboardInterrupt:
        # Pressing Ctrl+C to exit the program as before.
        print("Exiting...")


def emotion_specific_flow(story: list, emotion: str):
    client = OpenAI(api_key=openai_api_key)
    furhat = ExpressiveRemoteAPI(furhat_ip, openai=client)

    for message in story:
        furhat.react_to_text(message)
        furhat.say(text=message, blocking=True)


if __name__ == "__main__":
    # You can uncomment the following lines to enable additional log output.
    # import logging
    # logging.basicConfig(level=logging.INFO)
    expressive_flow()  # use typical_flow() if you want to compare with the default implementation
