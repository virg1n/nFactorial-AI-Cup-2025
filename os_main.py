import time
import json
import pyautogui

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

def main():
    done = False
    # aim="Open arbuz.kz in chrome and order chicken."
    # aim="Open arbuz.kz in chrome and order chicken"
    aim="Open my main vault in obsidian app"
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
    


if __name__ == '__main__':
    main()
