from openai import OpenAI
from omegaconf import OmegaConf
import json
import re

#GPT utils

class OpenAIClient():
    def __init__(self, prompt_file, max_history_length=6):
        self.system_message = OmegaConf.load(prompt_file).config.system_message
        self.start_message = {"role":"system", "content": self.system_message}
        self.message = {"role": "user", "content": ""}
        self.input = {"text": ""}
        self.message_history = []
        self.max_history_length = max_history_length
        prompt_config = OmegaConf.load(prompt_file).config
        self.client = OpenAI(api_key=prompt_config.api_key, organization = prompt_config.organization)
    
    def queryGPT(self, user_input):
        #print("Query GPT: "+ user_input) 
        self.message["content"] = user_input
        to_store = self.message.copy()
        if len(self.message_history) >= self.max_history_length:
            #print("clear message history")
            delete = self.message_history.pop(0)
        self.message_history.append(to_store)
        to_send = self.message_history
        to_send.insert(0, self.start_message)
        #print(to_send)
        try:
            response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=to_send,
            temperature=1,
            max_tokens=150)
            response_text = response.choices[0].message.content
            #print("Raw Response:", response_text)
            try:
                parsed_response = json.loads(response_text)
                if isinstance(parsed_response, dict) and "sentence" in parsed_response:
                    output_sentence = parsed_response["sentence"]
                else:
                    print("Unexpected response format:", parsed_response)
                    output_sentence = response_text
            except json.JSONDecodeError:
                print("Response is not valid JSON:", response_text)
                output_sentence = response_text 
            if len(self.message_history) >= self.max_history_length:
                self.message_history.pop(0)
            self.message_history.append({"role":"assistant", "content": response_text})
            #print(output_sentence)
            return output_sentence
        except Exception as e:
            print(f"Error in response: {e}")
            return "Errore nella risposta"
       
    
    def queryGPT_emotions (self, user_input):
        #print("Query GPT: "+ user_input) 
        self.message["content"] = user_input
        to_store = self.message.copy()
        if len(self.message_history) >= self.max_history_length:
            #print("clear message history")
            delete = self.message_history.pop(0)
        self.message_history.append(to_store)
        to_send = self.message_history
        to_send.insert(0, self.start_message)
        #print(to_send)
        try:
            response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=to_send,
            temperature=1,
            max_tokens=150)
            response_text = response.choices[0].message.content
            #print("Raw Response:", response_text)
            json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group()
                try:
                    parsed_response = json.loads(json_text)
                    if isinstance(parsed_response, dict) and "sentence" in parsed_response and "emotion" in parsed_response:
                        output_sentence = parsed_response["sentence"]
                        emotion = parsed_response["emotion"]
                        if emotion not in ["happy", "sad", "angry", "scared", "surprised", "disgusted", "neutral"]:
                            emotion = "neutral"
                    else:
                        print("JSON found but missing required fields, using raw response instead:", parsed_response)
                        output_sentence = response_text
                        emotion = "neutral"
                except json.JSONDecodeError:
                    print("Malformed JSON found, using raw response instead:", json_text)
                    output_sentence = response_text
                    emotion = "neutral"
            else:
                output_sentence = response_text
                emotion = "neutral"
            corrected_content = json.dumps({"sentence": output_sentence, "emotion": emotion}, ensure_ascii=False)
            if len(self.message_history) >= self.max_history_length:
                self.message_history.pop(0)
            self.message_history.append({"role":"assistant", "content": corrected_content})
            #print(output_sentence)
            return output_sentence, emotion
        except Exception as e:
            print(f"Error in response: {e}")
            return "Errore nella risposta", "neutral"
        
    def clear_conversation(self):
        self.message_history = []
