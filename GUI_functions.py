import pyautogui
from PIL import ImageGrab
import pytesseract
from PIL import ImageGrab, ImageDraw, ImageFont
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



def click_one_word_ocr(
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
        if debug:
                print(f"[OCR] '{w}' conf={conf} @ "
                  f"({data['left'][i]}, {data['top'][i]}, "
                  f"{data['width'][i]}×{data['height'][i]})")
        
        if target in wl:
           
            
            left, top = data['left'][i], data['top'][i]
            width, height = data['width'][i], data['height'][i]
            area = width * height
            # store (conf, area, top, left, bbox)
            candidates.append((conf, area, top, left, (left, top, width, height)))

    if not candidates:
        print(f"Text '{text}' not found with confidence ≥{min_confidence}")
        return(f"Text '{text}' not found with confidence ≥{min_confidence}")

    # sort by:
    # 1) confidence desc, 2) area desc (largest box first),
    # 3) top asc, 4) left asc
    candidates.sort(key=lambda x: (-x[1], -x[0], x[2], x[3]))

    # pick the best one
    _, _, _, _, (l, t, w, h) = candidates[0]
    cx = l + w // 2
    cy = t + h // 2

    pyautogui.click(cx, cy)
    if debug:
        print(f"Clicked '{text}' at ({cx},{cy}) from bbox {(l, t, w, h)}")


    return True


def click_multi_words_ocr(
    text: str,
    lang: str = 'eng',
    *,
    debug: bool = True,
    min_confidence: int = 50
):
    """
    Finds `text` on the screen via OCR (case-insensitive, substring match)
    by grouping words into lines and clicking the best match.
    """
    target = text.lower()
    img = ImageGrab.grab()
    data = pytesseract.image_to_data(img, lang=lang, output_type=Output.DICT)
    n = len(data['text'])

    # 1) Group word-indices by (block_num, line_num)
    lines = {}
    for i in range(n):
        w = data['text'][i].strip()
        if not w:
            continue
        conf = int(data['conf'][i] or -1)
        if conf < min_confidence:
            continue
        key = (data['block_num'][i], data['line_num'][i])
        lines.setdefault(key, []).append(i)
        if debug:
            print(f"[OCR] word='{w}' conf={conf} block,line={key} "
                  f"@({data['left'][i]}, {data['top'][i]}, "
                  f"{data['width'][i]}×{data['height'][i]})")

    # 2) For each line, build a phrase and see if it contains the target
    candidates = []
    for key, idxs in lines.items():
        # sort by X so words are in reading order
        idxs = sorted(idxs, key=lambda i: data['left'][i])
        phrase = " ".join(data['text'][i] for i in idxs).lower()

        if target in phrase:
            # build a bounding box around all these words
            lefts   = [data['left'][i] for i in idxs]
            tops    = [data['top'][i] for i in idxs]
            rights  = [data['left'][i] + data['width'][i]  for i in idxs]
            bottoms = [data['top'][i]  + data['height'][i] for i in idxs]

            l, t = min(lefts), min(tops)
            r, b = max(rights), max(bottoms)
            w, h = r - l, b - t

            # average confidence, area, y-position, x-position
            avg_conf = sum(int(data['conf'][i] or 0) for i in idxs) / len(idxs)
            area     = w * h

            candidates.append((avg_conf, area, t, l, (l, t, w, h)))
            if debug:
                print(f"[MATCH] phrase='{phrase}' -> bbox={(l,t,w,h)}, "
                      f"avg_conf={avg_conf:.1f}, area={area}")

    if not candidates:
        print(f"Text '{text}' not found with confidence ≥{min_confidence}")
        return(f"Text '{text}' not found with confidence ≥{min_confidence}")

    # 3) Pick best by: highest conf, then largest area, then top-most, then left-most
    candidates.sort(key=lambda x: (-x[0], -x[1], x[2], x[3]))
    _, _, _, _, (l, t, w, h) = candidates[0]

    # 4) Click the center
    cx, cy = l + w // 2, t + h // 2
    pyautogui.click(cx, cy)

    if debug:
        print(f"Clicked '{text}' at ({cx}, {cy}) from bbox {(l, t, w, h)}")

    return True

def take_screenshot(path):
    screenshot = pyautogui.screenshot()
    screenshot.save(path)

# import pyautogui
# from PIL import ImageGrab, ImageDraw, ImageFont

# def show_grid_overlay(
#     rows: int,
#     cols: int,
#     save_path: str = "grid_overlay.png",
#     line_color: str = "red",
#     line_width: int = 2,
#     font_size: int = 40,
#     font_color: str = "red"
# ) -> None:
#     """
#     Grab the full screen, draw an rows×cols grid on top, number each cell
#     from 0 to rows*cols-1 (row-major), and save to disk.
#     """
#     img = ImageGrab.grab().convert("RGB")
#     w, h = img.size
#     draw = ImageDraw.Draw(img)
#     font = ImageFont.load_default()

#     # draw vertical & horizontal grid lines
#     for c in range(1, cols):
#         x = c * w / cols
#         draw.line([(x, 0), (x, h)], fill=line_color, width=line_width)
#     for r in range(1, rows):
#         y = r * h / rows
#         draw.line([(0, y), (w, y)], fill=line_color, width=line_width)

#     # label each cell
#     cell_w = w / cols
#     cell_h = h / rows
#     for r in range(rows):
#         for c in range(cols):
#             idx = r * cols + c
#             # center of this cell
#             cx = (c + 0.5) * cell_w
#             cy = (r + 0.5) * cell_h
#             label = str(idx)
#             text_w, text_h = draw.textsize(label, font=font)
#             # offset to truly center the text
#             tx = cx - text_w / 2
#             ty = cy - text_h / 2
#             draw.text((tx, ty), label, fill=font_color, font=font)

#     img.save(save_path)
#     print(f"Saved {rows}×{cols} numbered grid overlay to {save_path}.")


# def click_grid_cell(
#     cell_index: int,
#     rows: int,
#     cols: int
# ) -> None:
#     """
#     Click the center of a cell by its 0-based index in a rows×cols grid.
    
#     Args:
#       cell_index: which cell to click, from 0 up to rows*cols-1 (row-major order)
#       rows: total number of rows in the grid
#       cols: total number of columns in the grid
#     """
#     # turn index → 1-based row & col
#     row = (cell_index // cols) + 1
#     col = (cell_index %  cols) + 1

#     screen_w, screen_h = pyautogui.size()
#     cell_w = screen_w  / cols
#     cell_h = screen_h / rows

#     # center point:
#     x = (col - 0.5) * cell_w
#     y = (row - 0.5) * cell_h

#     pyautogui.click(int(x), int(y))
#     print(f"Clicked cell #{cell_index} → (row={row},col={col}) at ({int(x)},{int(y)})")



import easyocr
import pyautogui
from PIL import ImageGrab
import cv2
import numpy as np
from PIL import ImageGrab

_READER = easyocr.Reader(['en'], gpu=True)

def click_easyocr_one_word(
    text: str,
    debug: bool = True,
    min_confidence: float = 0.4
) -> bool:
    """
    Finds a substring `text` in any OCR result, but tries largest
    then highest-confidence boxes first.
    """
    # 1) grab screen & prep for EasyOCR
    img_pil  = ImageGrab.grab()
    img_np   = np.array(img_pil)  
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # 2) run OCR
    raw = _READER.readtext(img_gray)  # list of (bbox, string, conf)

    # 3) compute area & collect
    scored = []
    for bbox, detected, conf in raw:
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        area = w * h
        scored.append((area, conf, bbox, detected))

    # 4) sort by area DESC, then conf DESC
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    # 5) scan for your text
    for area, conf, bbox, detected in scored:
        if conf < min_confidence:
            continue
        if text.lower() in detected.lower():
            # center point
            cx = int(sum(pt[0] for pt in bbox) / 4)
            cy = int(sum(pt[1] for pt in bbox) / 4)
            if debug:
                print(f"[OCR] match='{detected}' conf={conf:.2f} area={area} → click=({cx},{cy})")
            pyautogui.click(cx, cy)
            return True

    if debug:
        print(f"No match for '{text}' with conf ≥{min_confidence}")
    return False


def click_easyocr_multi_words(
    text: str,
    debug: bool = True,
    min_confidence: float = 0.5
) -> bool:
    """
    Finds an *exact* multi-word `text` match, but tries the largest,
    highest-confidence boxes first.
    """
    target = text.lower().split()

    img_pil  = ImageGrab.grab()
    img_np   = np.array(img_pil)
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    raw = _READER.readtext(img_gray)

    scored = []
    for bbox, detected, conf in raw:
        words = detected.lower().split()
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        area = w * h
        scored.append((area, conf, bbox, words, detected))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    for area, conf, bbox, words, detected in scored:
        if conf < min_confidence:
            continue
        if words == target:
            cx = int(sum(pt[0] for pt in bbox) / 4)
            cy = int(sum(pt[1] for pt in bbox) / 4)
            if debug:
                print(f"[MATCH] '{detected}' conf={conf:.2f} area={area} → click=({cx},{cy})")
            pyautogui.click(cx, cy)
            return True

    if debug:
        print(f"No exact multi-word match for '{text}' with conf ≥{min_confidence}")
    return False