import pyautogui
from PIL import ImageGrab
import pytesseract
from pytesseract import Output
import cv2
import numpy as np

# Update this path to match where Tesseract is installed on your machine.
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Ensure Tesseract is installed and the path is set correctly.
# Example for Windows:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def click_sequence(keys: list, interval: float = 0.1):
    """
    Presses a combination of keys (e.g., ['ctrl','c']) or single key.
    """
    if len(keys) > 1:
        pyautogui.hotkey(*keys, interval=interval)
    else:
        pyautogui.press(keys[0])


def click_coords(x: int, y: int, clicks: int = 1, interval: float = 0.0, button: str = 'left'):
    """
    Moves mouse to (x,y) and clicks.
    """
    pyautogui.click(x=x, y=y, clicks=clicks, interval=interval, button=button)


def click_text_image(image_path: str, confidence: float = 0.8):
    """
    Finds the best match of the given template image on screen and clicks its center.
    
    Args:
        image_path (str): Path to the small screenshot of the text/button.
        confidence (float): Minimum normalized match score (0.0 to 1.0).
    
    Raises:
        Exception: If no match at or above confidence is found.
    """
    # 1) grab full screen and convert to BGR numpy array
    screen_pil = ImageGrab.grab()
    screen_np = cv2.cvtColor(np.array(screen_pil), cv2.COLOR_RGB2BGR)

    # 2) load template (our image snippet)
    template = cv2.imread(image_path)
    if template is None:
        raise Exception(f"Could not load template image: {image_path}")
    th, tw = template.shape[:2]

    # 3) do template matching
    result = cv2.matchTemplate(screen_np, template, cv2.TM_CCOEFF_NORMED)
    # find all locations above threshold
    ys, xs = np.where(result >= confidence)
    if len(xs) == 0:
        raise Exception(f"Image '{image_path}' not found on screen (conf >= {confidence})")

    # 4) collect (score, x, y) for each hit
    hits = [(result[y, x], x, y) for x, y in zip(xs, ys)]
    # sort by score desc, then y asc, then x asc
    hits.sort(key=lambda h: (-h[0], h[2], h[1]))

    # 5) pick the best hit and compute its center
    best_score, bx, by = hits[0]
    center_x = bx + tw // 2
    center_y = by + th // 2

    # 6) click it
    pyautogui.click(center_x, center_y)



def click_text_ocr(
    text: str,
    lang: str = 'eng',
    *,
    debug: bool = True,
    min_confidence: int = 50
):
    """
    Finds `text` on the screen via OCR (case-insensitive, substring match)
    and clicks the occurrence with the highest confidence, then largest size,
    then top-most, then left-most.
    """
    target = text.lower()
    img = ImageGrab.grab()
    data = pytesseract.image_to_data(img, lang=lang, output_type=Output.DICT)

    candidates = []
    # collect matches
    for i, word in enumerate(data['text']):
        w = word.strip()
        if not w:
            continue
        conf = int(data['conf'][i] or -1)
        wl = w.lower()
        
        if target in wl and conf >= min_confidence:
            if debug:
                print(f"[OCR] '{w}' conf={conf} @ "
                  f"({data['left'][i]}, {data['top'][i]}, "
                  f"{data['width'][i]}×{data['height'][i]})")
            
            left, top = data['left'][i], data['top'][i]
            width, height = data['width'][i], data['height'][i]
            area = width * height
            # store (conf, area, top, left, bbox)
            candidates.append((conf, area, top, left, (left, top, width, height)))

    if not candidates:
        raise Exception(f"Text '{text}' not found with confidence ≥{min_confidence}")

    # sort by:
    # 1) confidence desc, 2) area desc (largest box first),
    # 3) top asc, 4) left asc
    candidates.sort(key=lambda x: (-x[0], -x[1], x[2], x[3]))

    # pick the best one
    _, _, _, _, (l, t, w, h) = candidates[2]
    cx = l + w // 2
    cy = t + h // 2

    pyautogui.click(cx, cy)
    if debug:
        print(f"Clicked '{text}' at ({cx},{cy}) from bbox {(l, t, w, h)}")
