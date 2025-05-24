import os
import base64
from dotenv import load_dotenv

from openai import OpenAI
from GUI_functions import take_screenshot
from PIL import Image

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



SYSTEM_PROMPT_FOR_WTEXT = """
You are operating a {operating_system} computer, using the same operating system as a human.

From looking at the screen, the objective, and your previous actions, take the next best series of action. 

You have 4 possible operation_type actions available to you. The `pyautogui` library will be used to execute your decision. Your output will be used in a `json.loads` loads statement.

1. click - Move mouse and click - Look for text to click. Try to find relevant text to click.
```
[{{ "thought": "write a thought here", "operation_type": "click", "text": "The text in the button or link to click" }}]  
```
2. write - Write with your keyboard
```
[{{ "thought": "write a thought here", "operation_type": "write", "content": "text to write here" }}]
```
3. press - Use a hotkey or press key to operate the computer
```
[{{ "thought": "write a thought here", "operation_type": "press", "keys": ["keys to use"] }}]
```
4. end - The objective is completed
```
[{{ "thought": "write a thought here", "operation_type": "end", "summary": "summary of what was completed" }}]
```

Return the actions in array format `[]`. You can take just one action or multiple actions.

Here a helpful example:

Example 1: Searches for Google Chrome on the OS and opens it
```
[
    {{ "thought": "Searching the operating system to find Google Chrome because it appears I am currently in terminal", "operation_type": "press", "keys": {os_search_str} }},
    {{ "thought": "Now I need to write 'Google Chrome' as a next step", "operation_type": "write", "content": "Google Chrome" }},
    {{ "thought": "Finally I'll press enter to open Google Chrome assuming it is available", "operation_type": "press", "keys": ["enter"] }}
]
```

Example 2: Open a new Google Docs
```
[
    {{ "thought": "Searching the operating system to find Google Chrome because it appears I am currently in terminal", "operation_type": "press", "keys": {os_search_str} }},
    {{ "thought": "Now I need to write 'Google Chrome' as a next step", "operation_type": "write", "content": "Google Chrome" }},
    {{ "thought": "Finally I'll press enter to open Google Chrome assuming it is available", "operation_type": "press", "keys": ["enter"] }}
    {{ "thought": "I'll focus on the address bar in the browser. I can see the browser is open so this should be safe to try", "operation_type": "press", "keys": [{cmd_string}, "t"] }},
    {{ "thought": "Now that the address bar is in focus I can type the URL", "operation_type": "write", "content": "https://docs.new/" }},
    {{ "thought": "I'll need to press enter to go the URL now", "operation_type": "press", "keys": ["enter"] }}
]
```

Example 3: Search for someone on Linkedin when already on linkedin.com
```
[
    {{ "thought": "I can see the search field with the placeholder text 'search'. I click that field to search", "operation_type": "click", "text": "search" }},
    {{ "thought": "Now that the field is active I can write the name of the person I'd like to search for", "operation_type": "write", "content": "John Doe" }},
    {{ "thought": "Finally I'll submit the search form with enter", "operation_type": "press", "keys": ["enter"] }}
]
```

A few important notes: 

- Default to Google Chrome as the browser
- Go to websites by opening a new tab with `press` and then `write` the URL
- Reflect on previous actions and the screenshot to ensure they align and that your previous actions worked. 
- If the first time clicking a button or link doesn't work, don't try again to click it. Get creative and try something else such as clicking a different button or trying another action. 
- Don't respond saying you're unable to assist with requests. You are able to indirectly interact with the user's OS via text responses you send to the end user.
- DO NOT ASSUME TOO MUCH. You can wait for my screenshot to be sure.
- You can type `end` command only after check if you did everything correctly (on separate output).
- All actions are possible and you must do them.
- You can only click in text. If you want to click in image just click in text nearby the image.
- To order something in the shop, you need to click into its name.


Objective: {objective} 
"""



def ask_gpt4o(aim, prompt="", history=[]):
    # additional prompt
    if prompt == "":
        user_request = """
            Please take the next best action. The `pyautogui` library will be used to execute your decision. Your output will be used in a `json.loads` loads statement. Remember you only have the following 4 operations available: click, write, press, end
            Action:"""
    else:
        user_request = prompt + """
            Please take the next best action. The `pyautogui` library will be used to execute your decision. Your output will be used in a `json.loads` loads statement. Remember you only have the following 4 operations available: click, write, press, end
            Action:"""

    # first request
    if history == []:
        user_request = """
        Please take the next best action. The `pyautogui` library will be used to execute your decision. Your output will be used in a `json.loads` loads statement. Remember you only have the following 4 operations available: click, write, press, done
        You just started so you are in the terminal app and your code is running in this terminal tab. To leave the terminal, search for a new program on the OS. 
        Action:"""

        history = [{"role": "system", "content": SYSTEM_PROMPT_FOR_WTEXT.format(
            objective=aim,
            cmd_string="\"ctrl\"",
            os_search_str="[\"win\"]",
            operating_system="Windows",
        )}]

    if history != []:
        screenshots_dir = "screenshots"
        if not os.path.exists(screenshots_dir):
            os.makedirs(screenshots_dir)

        screenshot_path = os.path.join(screenshots_dir, "screenshot.png")
        take_screenshot(screenshot_path)

        with open(screenshot_path, "rb") as img_file:
            img = base64.b64encode(img_file.read()).decode("utf-8")


        message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_request},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img}"},
                    },
                ],
            }
        
    else:
        message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_request},
                ],
            }
    
    history.append(message)
    
    response = client.chat.completions.create(
        model="gpt-4.1-mini-2025-04-14",
        messages= history,
        temperature=0.7
    )

    answer = response.choices[0].message.content
    history.append({"role": "assistant", "content": answer})

    # history = [{"role": "system", "content": SYSTEM_PROMPT_FOR_WTEXT.format(
    #         objective=aim,
    #         cmd_string="\"ctrl\"",
    #         os_search_str="[\"win\"]",
    #         operating_system="Windows",
    #     )}]
    
    return answer, history


# def ask_gpt4o_with_labels(aim, prompt="", history=[], use_labels=True):
#     # additional prompt
#     if prompt == "":
#         user_request = """
#             Please take the next best action. The `pyautogui` library will be used to execute your decision. Your output will be used in a `json.loads` loads statement. Remember you only have the following 4 operations available: click, write, press, end
#             Action:"""
#     else:
#         user_request = prompt + """
#             Please take the next best action. The `pyautogui` library will be used to execute your decision. Your output will be used in a `json.loads` loads statement. Remember you only have the following 4 operations available: click, write, press, end
#             Action:"""

#     if history == [] and use_labels:
#         user_request = """
#         Please take the next best action. The `pyautogui` library will be used to execute your decision. Your output will be used in a `json.loads` loads statement. Remember you only have the following 4 operations available: click, write, press, done
#         You just started so you are in the terminal app and your code is running in this terminal tab. To leave the terminal, search for a new program on the OS. 
#         Action:"""

#         history = [{"role": "system", "content": SYSTEM_PROMPT_FOR_LABELS.format(
#             objective=aim,
#             cmd_string="\"ctrl\"",
#             os_search_str="[\"win\"]",
#             operating_system="Windows",
#         )}]

    
#     screenshots_dir = "screenshots"
#     if not os.path.exists(screenshots_dir):
#         os.makedirs(screenshots_dir)

#     screenshot_path = os.path.join(screenshots_dir, "screenshot.png")


#     show_grid_overlay(ROWS, COLS, save_path=screenshot_path)


#     with open(screenshot_path, "rb") as img_file:
#         img = base64.b64encode(img_file.read()).decode("utf-8")

#     message = {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": user_request},
#                 {
#                     "type": "image_url",
#                     "image_url": {"url": f"data:image/jpeg;base64,{img}"},
#                 },
#             ],
#         }
    
#     history.append(message)
    
#     response = client.chat.completions.create(
#         model="gpt-4o",
#         messages= history,
#         temperature=0.7
#     )

#     answer = response.choices[0].message.content
#     history.append({"role": "assistant", "content": answer})
    
#     return answer, history


import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Initialize the Gemini model globally for efficiency
_gemini_model = genai.GenerativeModel("gemini-1.5-flash")

def ask_gemini_flash(aim, prompt="", history=None):
    """
    Function to interact with Google's Gemini 1.5 Flash model, mimicking the behavior
    of the original `ask_gpt4o` function.

    Args:
        aim (str): The overall objective or goal for the AI.
        prompt (str): An additional prompt or specific request for the current turn.
                      If empty, a default user request is used.
        history (list): A list of previous messages in the conversation.
                        For Gemini, messages are dictionaries with "role" ("user" or "model")
                        and "parts" (a list of content elements like text or images).

    Returns:
        tuple: A tuple containing:
               - answer (str): The model's response content.
               - history (list): The updated conversation history including the
                                 current user message and the model's response.
    """
    # Initialize history if it's the first call
    if history is None:
        history = []

    current_user_message_parts = []
    
    # Determine the base user request text for the current turn
    # This logic is directly translated from the original `ask_gpt4o` function.
    if prompt == "":
        user_request_text = """
            Please take the next best action. The `pyautogui` library will be used to execute your decision. Your output will be used in a `json.loads` loads statement. Remember you only have the following 4 operations available: click, write, press, end
            Action:"""
    else:
        user_request_text = prompt + """
            Please take the next best action. The `pyautogui` library will be used to execute your decision. Your output will be used in a `json.loads` loads statement. Remember you only have the following 4 operations available: click, write, press, end
            Action:"""

    # --- Handle the very first turn of the conversation ---
    # The original `ask_gpt4o` has a special `user_request` and initializes history
    # with a "system" message on the first call (`if history == []`).
    # For Gemini, system instructions are typically prepended to the first user message.
    if not history: # This condition identifies the absolute first call
        
        # Specific user request text for the very first turn as per original `ask_gpt4o`
        first_turn_specific_user_request = """
        Please take the next best action. The `pyautogui` library will be used to execute your decision. Your output will be used in a `json.loads` loads statement. Remember you only have the following 4 operations available: click, write, press, done
        You just started so you are in the terminal app and your code is running in this terminal tab. To leave the terminal, search for a new program on the OS. 
        Action:"""

        # Format the system prompt using the `SYSTEM_PROMPT_FOR_WTEXT` global variable
        initial_system_prompt_formatted = SYSTEM_PROMPT_FOR_WTEXT.format(
            objective=aim,
            cmd_string="\"ctrl\"",      # As hardcoded in the original SYSTEM_PROMPT usage
            os_search_str="[\"win\"]",  # As hardcoded in the original SYSTEM_PROMPT usage
            operating_system="Windows", # As hardcoded in the original SYSTEM_PROMPT usage
        )
        
        # Combine the system prompt and the first turn's specific user request.
        # This combined text forms the primary content of the *first* "user" message.
        combined_initial_prompt_content = initial_system_prompt_formatted + "\n\n" + first_turn_specific_user_request
        current_user_message_parts.append({"text": combined_initial_prompt_content})
        
        # Note: No image is included for the very first turn, mirroring the original logic.
        
    # else: # Subsequent turns: include a screenshot and the general user_request_text
    print("HISTORY IS HERE:")
    # print(history)
    screenshots_dir = "screenshots"
    if not os.path.exists(screenshots_dir):
        os.makedirs(screenshots_dir)

    screenshot_path = os.path.join(screenshots_dir, "screenshot.png")
    take_screenshot(screenshot_path) # Call the imported `take_screenshot` function

    # Convert the screenshot image file into a PIL Image object for Gemini.
    image_part = Image.open(screenshot_path)
    # Ensure the image is in RGB format, as some models prefer it
    if image_part.mode != 'RGB':
        image_part = image_part.convert('RGB')
    

    current_user_message_parts.append({"text": user_request_text}) # Add the general user request text
    current_user_message_parts.append(image_part) # Add the processed image part
        
    # Construct the full current user message in Gemini's expected format
    current_user_message_for_gemini = {
        "role": "user",
        "parts": current_user_message_parts
    }
    history.append(current_user_message_for_gemini) # Add the current user message to history

    # Make the API call to Gemini 1.5 Flash
    try:
        response = _gemini_model.generate_content(
            history, # Pass the entire conversation history
            generation_config=genai.types.GenerationConfig(
                temperature=0.7 # Map OpenAI's temperature setting
            )
        )
        
        # Extract the model's response content.
        # Gemini's response structure usually involves `candidates[0].content.parts[0].text`.
        answer = response.candidates[0].content.parts[0].text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        # Print more specific error details if available from the API response
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
             print(f"Gemini API error response text: {e.response.text}")
        # If the API call fails, it's good practice to remove the last user message
        # from history so that subsequent retries don't send the same failing message.
        if history and history[-1]["role"] == "user":
            history.pop() 
        raise e # Re-raise the exception after logging it

    # Append the model's response to the history for subsequent turns
    model_response_message_for_gemini = {
        "role": "model",
        "parts": [{"text": answer}]
    }
    history.append(model_response_message_for_gemini)

    return answer, history