#!/usr/bin/env python3

import click
import torch
import speech_recognition as sr
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

class MyWhisperMic(WhisperMic):
    def listen_loop(self, dictate: bool = False):
        threading.Thread(target=self.transcribe_forever).start()
        results = []
        while True:
            result = self.result_queue.get()
            results.append(result)
            if dictate:
                self.keyboard.type(result)
            if "let's play the story game" in result.lower():
                self.story_game()
            else:
                print(result)
            threading.Thread(target=call_gpt3_5, args=(result,)).start()
        return results
    
    def story_game(self):
        print("Starting the story game...")
        story = [
            {"role": "system", "content": "You are playing the story game. You should continue the story with one sentence after the user every time. "},
        ]
        while True:
            # User's turn
            result = self.result_queue.get()
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

@click.command()
@click.option("--model", default="base", help="Model to use", type=click.Choice(["tiny","base", "small","medium","large","large-v2"]))
@click.option("--device", default=("cuda" if torch.cuda.is_available() else "cpu"), help="Device to use", type=click.Choice(["cpu","cuda","mps"]))
@click.option("--english", default=False, help="Whether to use English model",is_flag=True, type=bool)
@click.option("--verbose", default=False, help="Whether to print verbose output", is_flag=True,type=bool)
@click.option("--energy", default=300, help="Energy level for mic to detect", type=int)
@click.option("--dynamic_energy", default=False,is_flag=True, help="Flag to enable dynamic energy", type=bool)
@click.option("--pause", default=0.8, help="Pause time before entry ends", type=float)
@click.option("--save_file",default=False, help="Flag to save file", is_flag=True,type=bool)
@click.option("--loop", default=False, help="Flag to loop", is_flag=True,type=bool)
@click.option("--dictate", default=False, help="Flag to dictate (implies loop)", is_flag=True,type=bool)
@click.option("--mic_index", default=None, help="Mic index to use", type=int)
@click.option("--list_devices",default=False, help="Flag to list devices", is_flag=True,type=bool)

def main(model: str, english: bool, verbose: bool, energy:  int, pause: float, dynamic_energy: bool, save_file: bool, device: str, loop: bool, dictate: bool,mic_index:Optional[int],list_devices:bool) -> None:
    if list_devices:
        print("Possible devices: ",sr.Microphone.list_microphone_names())
        return
    mic = MyWhisperMic(model=model, english=english, verbose=verbose, energy=energy, pause=pause, dynamic_energy=dynamic_energy, save_file=save_file, device=device,mic_index=mic_index)
    if not loop:
        result = mic.listen()
        print("You said: " + result)
    else:
        results = mic.listen_loop(dictate=dictate)
        print("Results: ", results)

if __name__ == "__main__":
    main()
