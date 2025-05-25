import time
import json
import pyautogui
from io import BytesIO
import base64

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

BASE_URL = "https://n-fac-deploy.vercel.app"

def main():
    executor = CodeExecutor()
    tokenizer = BertTokenizerFast.from_pretrained('./model_output/checkpoint-95')
    model = BertForSequenceClassification.from_pretrained('./model_output/checkpoint-95')

    while True:
        user_input = input("\nEnter command: ").strip()

        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
        outputs = model(**inputs)
        predictions = int(outputs.logits.argmax(-1))

        if ("--pyt" in user_input):
            predictions = 1

        print(predictions)
        
        if user_input.lower() == 'exit':
            print("Goodbye! 👋")
            break
            
        if not user_input:
            continue
        
        if predictions == 0:
            done = False
            aim = user_input

            # Send initial request to /start_task
            try:
                response = requests.post(f"{BASE_URL}/start_task", json={"objective": aim})
                response.raise_for_status()
                data = response.json()
                task_id = data["task_id"]
                actions = data["actions"]
                # history = data["history"]
            except requests.exceptions.RequestException as e:
                print(f"Error starting task: {e}")
                continue
            except KeyError:
                print("Invalid response from server")
                continue

            while not done:
                last_click_failed = False
                failed_text = ""

                for action in actions:
                    if action['operation_type'] == "press":
                        click_sequence(action["keys"], interval=0.2)
                        time.sleep(1)
                    elif action['operation_type'] == "write":
                        for char in action["content"]:
                            pyautogui.write(char)
                        time.sleep(1)
                    elif action['operation_type'] == "click":
                        text_to_click = action['text']
                        if len(text_to_click.split(" ")) == 1:
                            output = click_easyocr_one_word(text_to_click)
                        else:
                            output = click_easyocr_one_word(' '.join(text_to_click.split()[:3]))
                        if output:
                            time.sleep(1)
                        else:
                            last_click_failed = True
                            failed_text = text_to_click
                            break
                    elif action['operation_type'] == "end":
                        print("Operation Done")
                        done = True
                        break

                if done:
                    break

                # Take screenshot and encode it
                screenshot = pyautogui.screenshot()
                buffered = BytesIO()
                screenshot.save(buffered, format="JPEG", quality=85)
                screenshot_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

                # Send request to /get_action
                
                try:
                    payload = {
                        "task_id": task_id,
                        "screenshot_base64": screenshot_base64,
                        "last_click_failed": last_click_failed,
                        "failed_text": failed_text,
                        # "history": history
                    }
                    response = requests.post(f"{BASE_URL}/get_action", json=payload)
                    response.raise_for_status()
                    data = response.json()
                    actions = data["actions"]
                    # history = data["history"]
                except requests.exceptions.RequestException as e:
                    print(f"Error getting next action: {e}")
                    break
                except KeyError:
                    print("Invalid response from server")
                    break

        elif predictions == 1:
            try:
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
                
                if "explanation" in response and response["explanation"]:
                    print("\n📝 Explanation:")
                    print("-" * 50)
                    print(response["explanation"])
                    print("-" * 50)

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