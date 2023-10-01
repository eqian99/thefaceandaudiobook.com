#!/usr/bin/env python3

import click
import torch
import speech_recognition as sr
from utils.find_pi_zero_share_names import *
from face_emotion_recog_hugging import *
from face_detection import detect_face
from typing import Optional
import openai 
import os 
import threading 
openai.api_key = os.getenv("OPENAI_API_KEY")

from whisper_mic.whisper_mic import WhisperMic
import requests
from pydub import AudioSegment
from pydub.playback import play
import io
import time
import queue

queue_lock = threading.Lock()
game_queue = queue.Queue()  # Define a separate queue for the game
password = 'facebook'


def face_and_em_detection():
    print('Running face and emotion detection...')
    ## Image detection
    copy_images()
    face_path = get_latest_image(directory="img/")
    ## Pass img handles
    name = detect_face(face_path)
    emotion = get_emotion_from_image(face_path)
    
    print(name, emotion)
    return name, emotion

def run_face_and_em_detection():
    while True:
        name, emotion = face_and_em_detection()
        if name and emotion:
            # Call ElevenLabs and wait for it to finish speaking
            text = f"{name} is feeling {emotion}"
            duration = call_elevenlabs(text)
            time.sleep(duration)

def call_gpt3_5(text):
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
            {"role": "system", "content": "rewrite this like a pirate"},
            {"role": "user", "content": text},
        ]
    )
    gpt_response = response['choices'][0]['message']['content']
    print("GPT-3.5 response: ")
    print(gpt_response)

    # Speak the response using ElevenLabs
    voice_id = "21m00Tcm4TlvDq8ikWAM"  # Replace with your chosen voice ID
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {"xi-api-key": os.getenv("ELEVENLABS_API_KEY")}
    data = {
        "text": gpt_response,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0,
            "similarity_boost": 0,
            "style": 0,
            "use_speaker_boost": True
        }
    }
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        # Load audio from the response content
        audio = AudioSegment.from_file(io.BytesIO(response.content), format="mp3")
        # Play the audio
        # play(audio)
    else:
        print("Failed to generate speech:", response.text)

import time

def call_elevenlabs(text):
    voice_id = "21m00Tcm4TlvDq8ikWAM"  # Replace with your chosen voice ID
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {"xi-api-key": os.getenv("ELEVENLABS_API_KEY")}
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0,
            "similarity_boost": 0,
            "style": 0,
            "use_speaker_boost": True
        }
    }
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        # Load audio from the response content
        audio = AudioSegment.from_file(io.BytesIO(response.content), format="mp3")
        # Play the audio
        play(audio)
        # Return the duration of the audio in seconds
        return len(audio) / 1000
    else:
        print("Failed to generate speech:", response.text)
        return 0

def story_game(game_queue, call_elevenlabs):
    def run_game():
        print("Starting the story game...")
        story = [
            {"role": "system", "content": """You are play a game with the user, 
            where you both take turns adding to a story. The user will start. 
            Every sentence will be added to the story, so make sure you only say 
            things that continue the story. Only reply one short sentence. """},
        ]
        while True:
            with queue_lock:  # Use with statement for acquiring and releasing the lock
                print(f"Queue size before get: {game_queue.qsize()}")  
                result = game_queue.get()  # Use game_queue instead of result_queue
                print(f"Queue size after get: {game_queue.qsize()}")
                print(f"storyloop listen_loop received: {result}")
                if "let's stop the story game" in result.lower():
                    print("Stopping the story game...")
                    break
                story.append({"role": "user", "content": result})
                print("User: " + result)

                # Model's turn
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=story
                )
                model_result = response['choices'][0]['message']['content']
                story.append({"role": "assistant", "content": model_result})
                print("Model: " + model_result)
                # Call ElevenLabs and wait for it to finish speaking
                duration = call_elevenlabs(model_result)
                time.sleep(duration)

    threading.Thread(target=run_game).start()

class MyWhisperMic(WhisperMic):

    def __init__(self, game_queue, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.game_queue = game_queue  # Store a reference to the game_queue
    
    def listen_loop(self, dictate: bool = False) -> None:
        threading.Thread(target=self.transcribe_forever).start()
        while True:
            self.game_queue = self.result_queue
            result = self.result_queue.get()
            #with queue_lock:  # Use with statement for acquiring and releasing the lock
            print(f"listen_loop received: {result}")
            if "story game" in result.lower():
                print("Starting the story game...")
                story_game(self.game_queue, call_elevenlabs)  # Pass game_queue to story_game
            elif dictate:
                self.keyboard.type(result)
            else:
                print(result)

@click.command()
@click.option("--model", default="base", help="Model to use", type=click.Choice(["tiny","base", "small","medium","large","large-v2"]))
@click.option("--device", default=("cuda" if torch.cuda.is_available() else "cpu"), help="Device to use", type=click.Choice(["cpu","cuda","mps"]))
@click.option("--english", default=False, help="Whether to use English model",is_flag=True, type=bool)
@click.option("--verbose", default=False, help="Whether to print verbose output", is_flag=True,type=bool)
@click.option("--energy", default=800, help="Energy level for mic to detect", type=int)
@click.option("--dynamic_energy", default=False,is_flag=True, help="Flag to enable dynamic energy", type=bool)
@click.option("--pause", default=1, help="Pause time before entry ends", type=float)
@click.option("--save_file",default=False, help="Flag to save file", is_flag=True,type=bool)
@click.option("--loop", default=False, help="Flag to loop", is_flag=True,type=bool)
@click.option("--dictate", default=False, help="Flag to dictate (implies loop)", is_flag=True,type=bool)
@click.option("--mic_index", default=None, help="Mic index to use", type=int)
@click.option("--list_devices",default=False, help="Flag to list devices", is_flag=True,type=bool)

def main(model: str, english: bool, verbose: bool, energy:  int, pause: float, dynamic_energy: bool, save_file: bool, device: str, loop: bool, dictate: bool,mic_index:Optional[int],list_devices:bool) -> None:
    if list_devices:
        print("Possible devices: ",sr.Microphone.list_microphone_names())
        return
    mic = MyWhisperMic(game_queue, model=model, english=english, verbose=verbose, 
                       energy=energy, pause=pause, dynamic_energy=dynamic_energy,
                       save_file=save_file, device=device, mic_index=mic_index)    
    face_thread = threading.Thread(target=run_face_and_em_detection)
    face_thread.start()

    listen_thread = threading.Thread(target=mic.listen_loop, args=(dictate,))
    listen_thread.start()
    if not loop:
        result = mic.listen()
        print("You said: " + result)
    else:
        results = mic.listen_loop(dictate=dictate)
        print("Results: ", results)
    

if __name__ == "__main__":
    main()
