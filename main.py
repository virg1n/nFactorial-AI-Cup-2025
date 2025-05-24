import time
from functions import (
    click_sequence,
    click_coords,
    click_text_image,
    click_text_ocr
)

def main():
    # 1. Example: Copy selected text (Ctrl+C)
    # click_sequence(['win', 'd'], interval=0.2)
    # time.sleep(1)

    # 2. Example: Click at explicit coordinates
    # click_coords(100, 200)

    # # 3. Example: Click using an image snippet of text or button
    # # Save a screenshot snippet "ok_button.png" first, then:
    # click_text_image('ok_button.png', confidence=0.9)

    # 4. Example: Click based on OCR-detected text
    click_text_ocr('youtube')

if __name__ == '__main__':
    main()
