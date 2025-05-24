import os
import base64
from dotenv import load_dotenv

from openai import OpenAI
from GUI_functions import take_screenshot

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
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

Example 2: Open a new Google Docs when the browser is already open
```
[
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

        history = [{"role": "system", "content": SYSTEM_PROMPT.format(
            objective=aim,
            cmd_string="\"ctrl\"",
            os_search_str="[\"win\"]",
            operating_system="Windows",
        )}]

    
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
    
    history.append(message)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages= history,
        temperature=0.7
    )

    answer = response.choices[0].message.content
    history.append({"role": "assistant", "content": answer})
    
    return answer, history


