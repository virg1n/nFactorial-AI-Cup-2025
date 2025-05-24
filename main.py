import time
import json
import pyautogui

from GUI_functions import (
    click_sequence,
    click_coords,
    click_text_image,
    click_one_word_ocr,
    click_multi_words_ocr
)

from LLM_functions import ask_gpt4o

from functions import clean_json

def main():
    done = False
    aim="Open chrome, in chrome open linkedin and find Suvernev Bogdan"
    add_prompt = ""
    while not done:
        answer, history = ask_gpt4o(aim, add_prompt, history=[])
        add_prompt = ""
        cleaned_answer = clean_json(answer)
        answer = json.loads(cleaned_answer)
        for action in answer:
            print(action)
            if action['operation_type'] == "press":
                click_sequence(action["keys"], interval=0.2)
                time.sleep(0.5)

            elif action['operation_type'] == "write":
                for char in action["content"]:
                    pyautogui.write(char)
                time.sleep(0.4)

            elif action['operation_type'] == "click":
                if len(action["text"].split(" ")) == 1:
                    output = click_one_word_ocr((action["text"]))
                else:
                    output = click_multi_words_ocr(' '.join(action["text"].split()[:3]))
                if (output == True):
                    time.sleep(1)
                else:
                    add_prompt = "Previous Operation wasnt done. Try another method"

            elif action['operation_type'] == "end":
                print("Operation Done")
                done = True
                break
        


   
   
    # # 3. Example: Click using an image snippet of text or button
    # # Save a screenshot snippet "ok_button.png" first, then:
    # click_text_image('ok_button.png', confidence=0.9)

    # 4. Example: Click based on OCR-detected text
    # click_text_ocr('youtube')

if __name__ == '__main__':
    main()
