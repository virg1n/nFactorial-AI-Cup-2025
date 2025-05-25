import time
import json
import pyautogui
import speech_recognition as sr

from GUI_functions import (
    click_sequence,
    click_coords,
    click_text_image,
    click_one_word_ocr,
    click_multi_words_ocr,
    click_easyocr_one_word,
    click_easyocr_multi_words
)

from LLM_functions import ask_gpt4o, ask_gemini_flash

from functions import clean_json
import os
import openai
from dotenv import load_dotenv
import ast
import sys
from typing import Optional, Dict, List
import logging
import re
from deep_translator import GoogleTranslator
from langdetect import detect
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter
import json
from pathlib import Path
import requests

from transformers import BertForSequenceClassification, BertTokenizerFast

from pc_controller import get_user_confirmation, CodeExecutor, CommandHistory
import pyttsx3

READ = True

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ALLOWED_MODULES = {
    # Core Python modules
    'os', 're', 'sys', 'json', 'datetime', 'pathlib', 'logging',
    # Data processing and visualization
    'numpy', 'pandas', 'matplotlib', 'PIL', 'cv2',
    # File operations
    'shutil', 'glob', 'zipfile', 'csv',
    # String operations
    'string', 'collections', 'itertools',
    # Math and scientific
    'math', 'random', 'statistics',
    # Image processing
    'imageio', 'skimage',
    # File format handling
    'json', 'yaml', 'xml', 'csv',
    # Network and web
    'requests', 'urllib',
    # Text to speech
    'pyttsx3'
}


def get_voice_command():
    """Get voice command from microphone"""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\nListening... (speak your command)")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            print("Processing speech...")
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text.strip()
        except sr.WaitTimeoutError:
            print("No speech detected within timeout")
            return ""
        except sr.UnknownValueError:
            print("Could not understand the audio")
            return ""
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return ""

def main():
    executor = CodeExecutor()
    tokenizer = BertTokenizerFast.from_pretrained('./model_output/checkpoint-95')
    model = BertForSequenceClassification.from_pretrained('./model_output/checkpoint-95')

    print("\nVoice commands are now enabled!")
    print("You can type your command or say 'voice command' to use voice input")
    print("Say 'exit' or type 'exit' to quit")

    while True:
        print("\nEnter command (or say 'voice command'): ")
        user_input = input().strip()

        if user_input.lower() == "voice command":
            user_input = get_voice_command()
            if not user_input:
                continue

        if user_input.lower() == 'exit':
            if READ:
                executor.speak_text("Goodbye!")
            print("Goodbye! 👋")
            break
            
        if not user_input:
            continue

        # Load the tokenizer
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
        outputs = model(**inputs)
        predictions = int(outputs.logits.argmax(-1))

        if ("--pyt" in user_input):
            predictions = 1
        
        print(predictions)
        
        if predictions == 0:
            done = False
            # aim="Open my main vault in obsidian app"
            aim = user_input
            add_prompt = ""
            history = []
            while not done:

                # answer, history = ask_gpt4o(aim, add_prompt, history=[])
                
                answer, history = ask_gemini_flash(aim, add_prompt, history=history)
                
                cleaned_answer = clean_json(answer)
                try:
                    answer = json.loads(cleaned_answer)
                except:
                    answer, history = ask_gemini_flash(aim, add_prompt, history=history)
                    cleaned_answer = clean_json(answer)
                    answer = json.loads(cleaned_answer)
                    
                print(answer)
                for action in answer:
                    # print(action)
                    if action['operation_type'] == "press":
                        click_sequence(action["keys"], interval=0.2)
                        time.sleep(1)
                        add_prompt = ""

                    elif action['operation_type'] == "write":
                        for char in action["content"]:
                            pyautogui.write(char)
                        time.sleep(1)
                        add_prompt = ""

                    elif action['operation_type'] == "click":
                        if len(action["text"].split(" ")) == 1:
                            # output = click_one_word_ocr((action["text"]))
                            output = click_easyocr_one_word(action['text'])
                        else:
                            # output = click_multi_words_ocr(' '.join(action["text"].split()[:3]))
                            output = click_easyocr_one_word(' '.join(action["text"].split()[:3]))
                        if (output == True):
                            time.sleep(1)
                            add_prompt = ""
                        else:
                            add_prompt = f"Clicking onto {action['text']} failed. Try another method or another text. Everything that was after clicking aborded"
                            break

                    elif action['operation_type'] == "end":
                        print("Operation Done")
                        done = True
                        break

        elif predictions == 1:
            try:
                # user_input = input("\nEnter your command: ").strip()
                
                # Handle special commands
                if user_input.lower() == 'history':
                    last_commands = executor.history.get_last_n_commands()
                    print("\nLast commands:")
                    for cmd in last_commands:
                        status = "✅" if cmd["success"] else "❌"
                        print(f"{status} [{cmd['timestamp']}] {cmd['command']}")
                    continue

                if user_input.lower().startswith('search '):
                    query = user_input[7:].strip()
                    results = executor.history.search_commands(query)
                    print(f"\nSearch results for '{query}':")
                    for cmd in results:
                        status = "✅" if cmd["success"] else "❌"
                        print(f"{status} [{cmd['timestamp']}] {cmd['command']}")
                    continue
                    
                print("\nGenerating response...")
                response = executor.generate_response(user_input)
                
                # If there's an explanation, show it first
                if "explanation" in response and response["explanation"]:
                    print("\n📝 Explanation:")
                    print("-" * 50)
                    print(response["explanation"])
                    print("-" * 50)

                # If there's code to execute
                if "code" in response and response["code"]:
                    print("\nGenerated code:")
                    print("-" * 50)
                    print(response["code"])
                    print("-" * 50)
                    
                    if executor.validate_code(response["code"]):
                        if get_user_confirmation(response["code"]):
                            print("\nExecuting code...")
                            success = executor.execute_code(response["code"])
                            if success:
                                print("✅ Command executed successfully!")
                                executor.history.add_command(user_input, response["code"], True)
                            else:
                                error_msg = "Error during execution"
                                print(f"❌ {error_msg}")
                                executor.history.add_command(user_input, response["code"], False, error_msg)
                        else:
                            print("⚠️ Code execution cancelled by user")
                            executor.history.add_command(user_input, response["code"], False, "Cancelled by user")
                    else:
                        error_msg = "Code validation failed - potentially unsafe operations detected"
                        print(f"❌ {error_msg}")
                        executor.history.add_command(user_input, response["code"], False, error_msg)
                elif not response.get("explanation"):
                    error_msg = "Failed to generate response"
                    print(f"❌ {error_msg}")
                    executor.history.add_command(user_input, "", False, error_msg)
                    
            except KeyboardInterrupt:
                if READ:
                    executor.speak_text("Goodbye!")
                print("\nGoodbye! 👋")
                break
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Error: {error_msg}")
                executor.history.add_command(user_input, "", False, error_msg)



if __name__ == '__main__':
    main()
