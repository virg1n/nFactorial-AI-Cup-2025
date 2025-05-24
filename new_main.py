import time
import json
import pyautogui

from GUI_functions import (
    click_sequence,
    click_coords,
    click_text_image,
    click_one_word_ocr,
    click_multi_words_ocr,
    click_grid_cell
)

from LLM_functions import ask_gpt4o, ask_gpt4o_with_labels

from functions import clean_json

ROWS, COLS = 25, 15

def main():
    done = False
    aim="Open chrome, in chrome open linkedin and find Arman Suleimenov"
    add_prompt = ""
    while not done:
        answer, history = ask_gpt4o_with_labels(aim, add_prompt, history=[])
        add_prompt = ""
        cleaned_answer = clean_json(answer)
        answer = json.loads(cleaned_answer)
        for action in answer:
            print(action)
            if action['operation'] == "press":
                click_sequence(action["keys"], interval=0.2)
                time.sleep(0.5)

            elif action['operation'] == "write":
                for char in action["content"]:
                    pyautogui.write(char)
                time.sleep(0.4)

            elif action['operation'] == "click":
                click_grid_cell(int(action["label"]), ROWS, COLS)

            elif action['operation'] == "done":
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
