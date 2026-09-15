import os
import threading
import socket
import time
import asyncio
import random

from dotenv import load_dotenv
from omegaconf import OmegaConf

from SpeechSpeakerRecognizer import Recognizer
from OpenAIClient import OpenAIClient


import json
import xml.etree.ElementTree as ET

load_dotenv()
key = os.environ.get('SPEECH_KEY')
region = os.environ.get('SPEECH_REGION')
STORY_ID = "FAM" #"FAM" "B" "C" "END" select the activity ID
DEBUG = False    #True or False
TYPE = "ext" #local or ext for stt
MIC_IP = "10.186.13.11" 
ROBOT = "navel" #navel or other(for local tts)
if ROBOT == "navel":
    import navel
else:
    from Synthetizer import Synthetizer


with open('stories.json', 'r', encoding='utf-8') as f:
    stories = json.load(f)

story = next(s for s in stories["stories"] if s["id"] == STORY_ID)
filler_sentences = ["Fammi pensare un attimo a una risposta", "vi ho sentito, eleboro una risposta", "ho bisogno di pensare un attimo a cosa dire", "penso un attimo a cosa poter rispondere", "sto pensando a una risposta da darvi"]
characters_dict = {
    "N": "Navel",
    "S1": "Studente 1",
    "S2": "Studente 2",
    "I": "Insegnante"
}


class DialogueManager():
    def __init__(self):
        self.background_conversation = []
        #LLM client and prompt initialization
        prompt_file = "prompt.yaml"
        self.openai_client = OpenAIClient(prompt_file="prompt.yaml")
        if STORY_ID == "FAM":
            self.openai_client.system_message = OmegaConf.load(prompt_file).config.system_message_fam
        elif STORY_ID == "END":
            self.openai_client.system_message = OmegaConf.load(prompt_file).config.system_message_end
        elif STORY_ID == "DEMO":
            self.openai_client.system_message = OmegaConf.load(prompt_file).config.system_message_demo
        else:
            self.openai_client.system_message = self.openai_client.system_message.replace("{NAVEL_STORY_DESCRIPTION}", story["navel_story_description"])
            self.openai_client.system_message = self.openai_client.system_message.replace("{CURRENT_STORY}", story["current_story"])
        self.openai_client.start_message["content"] = self.openai_client.system_message
        #TTS initialization
        if ROBOT != "navel":
            self.tts = Synthetizer(lang="it-IT", key=key, region=region)

        self.speech_lock = asyncio.Lock()
        self.socket_ip = '127.0.0.1'
        self.socket_port = 9090
        #socket for Dialogue Manager to receive data from STT recognizer
        self.microphone_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.robot = None  # Persistent robot instance
    
    async def robot_say(self, text, emotion=None):
        #print(f"Robot says: {text}")
        if DEBUG:
            print(f"Robot would say: {text} with emotion: {emotion}")
            return
    
        async with self.speech_lock:
            if ROBOT != "navel":
                # Synthetizer.speak() uses .get() which waits for audio to complete
                self.tts.speak(text)
                print("Robot EMOTION: ", emotion)
            else:
                start_time = time.time()
                print(f"[{start_time}] NAVEL ROBOT SAYS: {text} AND SET FACE: {emotion}")
                if self.robot is None:
                    self.robot = await navel.Robot().__aenter__()
                #neutral: float = 0, happy: float = 0, sad: float = 0, surprise: float = 0, anger: float = 0, smile: float = 0
                if emotion and emotion.lower()=="happy":
                    self.robot.head_facial_expression(0,0.7,0,0,0,0.3)
                elif emotion and emotion.lower()=="neutral":
                    self.robot.head_facial_expression(1,0,0,0,0,0)
                elif emotion and emotion.lower() in ["surprised"]:
                    self.robot.head_facial_expression(0,0,0,1,0,0)
                elif emotion and emotion.lower() in ["fear", "scared"]:
                    self.robot.head_facial_expression(0,0,0.3,0.7,0,0)
                elif emotion and emotion.lower()=="sad":
                    self.robot.head_facial_expression(0,0,1,0,0,0)
                elif emotion and emotion.lower() in ["angry"]:
                    self.robot.head_facial_expression(0,0,0,0,1,0)
                elif emotion and emotion.lower() == "disgust":
                    self.robot.head_facial_expression(0,0.05, 0.12, 0.83,0)
                else:
                    self.robot.head_facial_expression(1,0,0,0,0,0)
                # robot.say() is async and waits for speech to complete
                print(f"[{time.time()}] NAVEL ROBOT STARTS SPEAKING after : {time.time()-start_time}")
                start_speech_time = time.time()
                await self.robot.say("<lang,it1>"+text)
                print(f"[{time.time()}] NAVEL ROBOT FINISHES SPEAKING after: {time.time()-start_speech_time}")

    def parse(self, text, character):
        # Parse the text and character to create a prompt for the LLM
        to_send = f"Background conversation: {self.background_conversation}\nCharacter: {character}\nUser input: {text}\n"
        self.background_conversation  = []
        return to_send
    
    def extract_speaker_and_sentence(self,xml_string):
        root = ET.fromstring(xml_string)
        profile = root.find("profile_id")
        speaker_id = profile.attrib.get("value")
        # Get only the text node (sentence), strip whitespace/newlines
        sentence = (profile.text or "").strip()
        return sentence
    
    async def dialogue(self):
        #Connect to microphone socket
        await self.robot_say(story["start_sentence"])
        input ("Press Enter to start the dialogue...")
        self.microphone_socket.connect((self.socket_ip, self.socket_port))
        #send ready signal to stt recognizer
        self.microphone_socket.send(b'ready')
        running = True
        while running:
            #Receive data from stt recognizer
            data = self.microphone_socket.recv(1024)
            received_data = data.decode('utf-8')
            if not data:
                break
            elif received_data == "timeout":
                print("STT recognizer timeout, please try again.")
                self.microphone_socket.send(b'ready')
                continue
            else:
                #extact the text from the received data (which is in XML format)
                if STORY_ID != "DEMO":
                    to_respond  = input(f"{received_data}\nThe robot need to respond to the user? [y/n]: ")
                else:
                    to_respond = 'y'

                if to_respond != 'n':
                    #say filler sentence to user in background task
                    t_filler = asyncio.create_task(self.robot_say(random.choice(filler_sentences)))
                    text = self.extract_speaker_and_sentence(received_data)
                    #character = input("Which character? [N/S1/S2/I]: ")
                    #create prompt with background conversation and dialogue history
                    #gpt_request = self.parse(text, characters_dict[character])
                    #send prompt to LLM and get response
                    #response = self.openai_client.queryGPT(text)
                    response, emotion = await asyncio.to_thread(self.openai_client.queryGPT_emotions, text)
                    print(f"[{time.time()}] LLM response: {response} LLM emotion: {emotion}")
                    #wait for filler speech to finish before speaking the response
                    await t_filler
                    await self.robot_say(response, emotion=emotion)
                #else:
                    #add to the background conversation
                    #self.background_conversation.append(received_data)
                    #print("Added to background conversation.")
            ending = input("close the dialogue ? [y/n]:")
            if ending == 'y':
                await self.robot_say(story["end_sentence"])
                running = False
                break
                
            elif STORY_ID != "FAM" and STORY_ID != "END" and STORY_ID != "DEMO":
                clear = input("clear the conversation history ? [y/n]:")
                if clear == 'y':
                    self.openai_client.clear_conversation()
                    self.background_conversation = []
            self.microphone_socket.send(b'ready')
        #Close and exit
        self.microphone_socket.close()

                    
class DebugRecognizer():
    #create a socket on port 9090 to send data to dialogue manager
    #This class is used for debugging purposes, to simulate the STT recognizer and send data to the dialogue manager without actually using the Azure Speech SDK.
    #Instead of recognizing speech, it will read lines from the console and send them to the dialogue manager.
    def __init__(self, socket_ip='127.0.0.1', socket_port=9090):
        self.socket_ip = socket_ip
        self.socket_port = socket_port
        self.alive = False
        self.microphone_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.microphone_socket.bind((self.socket_ip, self.socket_port))
        self.microphone_socket.listen(1)
        print(f"Debug Recognizer listening on {self.socket_ip}:{self.socket_port}")
    
    def listen(self):
        conn, addr = self.microphone_socket.accept()
        print(f"Connected by {addr}")
        conn.recv(1024)  # Wait for 'ready' signal from dialogue manager
        self.alive = True
        while self.alive:
            user_input = input("Enter simulated user speech (or 'timeout' to simulate timeout): ")
            if user_input == "close":
                break
            speaker = "unknown"
            xml_text = Recognizer._wrap_xml(user_input, user_id=speaker)  # Wrap the input in XML format
            conn.send(xml_text.encode('utf-8'))
            conn.recv(1024)  # Wait for acknowledgement from dialogue manager
        conn.close()
        self.close()
    
    def close(self):
        self.alive = False
        self.microphone_socket.close()
    
if __name__ == "__main__":
    dialogue_manager = DialogueManager()
    if DEBUG:
        stt = DebugRecognizer()
        t_sst = threading.Thread(target=stt.listen) 
        t_sst.start()
    else: 
        if TYPE == "local":
            stt = Recognizer(language="it-IT", azure_key=key, azure_region=region, emoACT_active=False, enable_speaker_recognition=False) 
            t_sst = threading.Thread(target=stt.listen) 
            t_sst.start()
        else: 
            dialogue_manager.socket_ip = MIC_IP
        
        if ROBOT == "navel":
            with navel.Robot() as robot:
                robot.config_set("cns_fer_cont_t1", 0, navel.DataType.U32)
                robot.config_set("cns_fer_cont_t2", 0, navel.DataType.U32)
                robot.config_set("cns_fer_peak_t1", 0, navel.DataType.U32)
                robot.config_set("cns_fer_peak_t2", 0, navel.DataType.U32)
                robot.config_set("cns_fer_overlay_inc", 0, navel.DataType.F32)
                robot.config_set("cns_fer_overlay_max", 0, navel.DataType.F32)
                robot.config_set("cns_fer_overlay_peak_min", 0, navel.DataType.F32)
                robot.config_set("cns_fer_peak_max", 0, navel.DataType.F32)
    asyncio.run(dialogue_manager.dialogue())
   
