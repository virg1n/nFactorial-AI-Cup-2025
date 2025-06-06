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
import pyttsx3


READ = True
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("Please set OPENAI_API_KEY in your .env file")

# Whitelist of allowed modules based on installed packages
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

class CommandHistory:
    def __init__(self, history_file: str = "command_history.json"):
        self.history_file = history_file
        self.history: List[Dict] = self.load_history()
        self.translator = GoogleTranslator(source='auto', target='en')

    def translate_to_english(self, text: str) -> str:
        """Translate text to English if it's not already in English."""
        try:
            detected_lang = detect(text)
            if detected_lang == 'en':
                return text
            translation = self.translator.translate(text)
            return translation
        except Exception as e:
            logging.error(f"Translation error in history: {e}")
            return text

    def load_history(self) -> List[Dict]:
        """Load command history from file."""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logging.error(f"Error loading history: {e}")
        return []

    def save_history(self):
        """Save command history to file."""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Error saving history: {e}")

    def add_command(self, command: str, generated_code: str, success: bool, error: Optional[str] = None):
        """Add a command to history, ensuring it's in English."""
        # Translate command to English
        english_command = self.translate_to_english(command)
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "command": english_command,
            "original_command": command,
            "generated_code": generated_code,
            "success": success,
            "error": error
        }
        self.history.append(entry)
        self.save_history()

    def get_last_n_commands(self, n: int = 5) -> List[Dict]:
        """Get the last n commands from history."""
        return self.history[-n:]

    def search_commands(self, query: str) -> List[Dict]:
        """Search command history for a query."""
        query_en = self.translate_to_english(query)
        return [
            entry for entry in self.history
            if query_en.lower() in entry["command"].lower()
        ]

    def get_successful_examples(self, n: int = 3) -> str:
        """Get last n successful commands as context for GPT."""
        successful = [cmd for cmd in self.history if cmd["success"]][-n:]
        if not successful:
            return ""
        
        examples = []
        for cmd in successful:
            examples.append(f"User request: {cmd['command']}\nGenerated code:\n{cmd['generated_code']}\n")
        
        return "\n".join(examples)

class CodeExecutor:
    def __init__(self):
        self.client = openai.OpenAI()
        self.translator = GoogleTranslator(source='auto', target='en')
        self.history = CommandHistory()
        # Initialize text-to-speech engine
        self.tts_engine = pyttsx3.init()
        
        # Get available voices and set a more natural voice
        voices = self.tts_engine.getProperty('voices')
        # Try to find a natural-sounding voice (usually the second voice is better)
        if len(voices) > 1:
            self.tts_engine.setProperty('voice', voices[1].id)
        
        # Configure TTS properties for more natural speech
        self.tts_engine.setProperty('rate', 175)  # Slightly faster, more natural pace
        self.tts_engine.setProperty('volume', 0.9)
        
    def translate_to_english(self, text: str) -> str:
        """Translate text from Russian to English."""
        try:
            # Detect the language
            detected_lang = detect(text)
            
            # If the text is already in English, return it as is
            if detected_lang == 'en':
                return text
                
            # Translate to English
            translation = self.translator.translate(text)
            logging.info(f"Translated from {detected_lang}: {translation}")
            return translation
        except Exception as e:
            logging.error(f"Translation error: {str(e)}")
            return text  # Return original text if translation fails

    def get_allowed_modules(self) -> Dict:
        """Get dictionary of allowed modules and their instances."""
        modules = {}
        for module_name in ALLOWED_MODULES:
            try:
                if module_name in sys.modules:
                    modules[module_name] = sys.modules[module_name]
                else:
                    module = __import__(module_name)
                    modules[module_name] = module
            except ImportError:
                continue
        return modules
        
    def generate_response(self, user_request: str) -> Dict[str, Optional[str]]:
        """Generate Python code and/or explanatory text based on user's natural language request."""
        try:
            # Translate request to English if needed
            english_request = self.translate_to_english(user_request)
            
            # Add available modules to the prompt
            available_modules = ", ".join(sorted(ALLOWED_MODULES))
            
            # Get successful examples from history
            examples = self.history.get_successful_examples()
            
            system_content = f"""You are a helpful assistant that can generate both Python code and explanatory text.
            When responding, you should:
            1. If the request requires code execution:
               - Generate Python code without any explanation or markdown formatting
               - Provide a brief summary of what the code will do
               - Format your response as JSON with "code" and "explanation" fields
            2. If the request is just for information:
               - Provide a detailed explanation or answer
               - Format your response as JSON with only the "explanation" field
            
            For code generation, follow these rules:
            - Only use these allowed modules: {available_modules}
            - Include all necessary imports from the allowed modules
            - You can use requests, but do not use anything that requires API Keys
            - Do not include ```python or ``` markers
            - When user wants to update notes, use: r"C:\\Users\\bogda\\Documents\\nFacObs\\Notes.md"
            - When user wants to update TODOs, use: r"C:\\Users\\bogda\\Documents\\nFacObs\\Todos.md"

            For matplotlib plots:
            1. Use plt.savefig() instead of plt.show()
            2. Save plots to a file in the current directory
            3. Print the path of the saved plot
            4. Close the figure with plt.close() to free memory"""

            messages = [
                {"role": "system", "content": system_content}
            ]
            
            # Add successful examples as context if available
            if examples:
                messages.append({
                    "role": "system",
                    "content": f"Here are some successful examples:\n{examples}"
                })
            
            messages.append({
                "role": "user",
                "content": f"Generate response for the following request: {english_request}"
            })
            
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                temperature=0.1,
                max_tokens=1000,
                response_format={ "type": "json_object" }
            )
            
            # Parse the JSON response
            response_data = json.loads(response.choices[0].message.content.strip())
            return response_data
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return {"explanation": f"Error: {str(e)}", "code": None}

    def clean_code(self, code: str) -> str:
        """Remove markdown formatting and clean the code."""
        # Remove markdown Python code blocks if present
        code = re.sub(r'```python\n', '', code)
        code = re.sub(r'```\n?', '', code)
        return code.strip()

    def validate_code(self, code: str) -> bool:
        """Validate the generated code for safety and syntax."""
        try:
            # Clean the code first
            cleaned_code = self.clean_code(code)
            
            # Parse the code to check for syntax errors
            ast.parse(cleaned_code)
            
            # List of forbidden operations
            forbidden = ['eval', 'exec', 'subprocess', 'os.system', 'input']
            
            # Check for forbidden operations
            for item in forbidden:
                if item in cleaned_code:
                    logging.warning(f"Forbidden operation found: {item}")
                    return False
            
            # Check that only allowed modules are imported
            tree = ast.parse(cleaned_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        if name.name.split('.')[0] not in ALLOWED_MODULES:
                            logging.warning(f"Forbidden module import: {name.name}")
                            return False
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.split('.')[0] not in ALLOWED_MODULES:
                        logging.warning(f"Forbidden module import: {node.module}")
                        return False
                    
            return True
        except Exception as e:
            logging.error(f"Code validation error: {str(e)}")
            return False

    def speak_text(self, text: str, is_execution_output: bool = False):
        """Speak the given text using text-to-speech."""
        if READ:
            try:
                # Only speak if it's execution output or if is_execution_output is False
                if not is_execution_output or (is_execution_output and text.strip()):
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
            except Exception as e:
                logging.error(f"Text-to-speech error: {str(e)}")

    def execute_code(self, code: str) -> bool:
        """Execute the generated and validated code."""
        try:
            # Clean the code before execution
            cleaned_code = self.clean_code(code)
            
            # Get allowed modules
            modules = self.get_allowed_modules()
            
            # Add built-ins and modules to the execution namespace
            exec_globals = {"__builtins__": __builtins__}
            exec_globals.update(modules)
            
            # Create a new local namespace
            local_namespace = {}

            # Redirect stdout to capture print output
            import io
            from contextlib import redirect_stdout
            output_buffer = io.StringIO()
            
            with redirect_stdout(output_buffer):
                # Execute the code in the isolated namespace
                exec(cleaned_code, exec_globals, local_namespace)
            
            # Get the captured output
            output = output_buffer.getvalue().strip()
            
            # Speak only the execution output
            if output:
                print("🔊 Output:", output)
                self.speak_text(output, is_execution_output=True)
            
            return True
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Code execution error: {error_msg}")
            self.speak_text(f"Error in code execution: {error_msg}", is_execution_output=True)
            return False

def get_user_confirmation(code: str) -> bool:
    """Get user confirmation before executing code."""
    print("\nDo you want to execute this code? Please review it carefully:")
    print("-" * 50)
    print(code)
    print("-" * 50)
    while True:
        response = input("Execute this code? (yes/no): ").lower().strip()
        if response in ['yes', 'y']:
            return True
        if response in ['no', 'n']:
            return False
        print("Please answer 'yes' or 'no'")

def main():
    executor = CodeExecutor()
    
    print("You can write commands in Russian or English!")
    print("Special commands:")
    print("- 'history': Show last 5 commands")
    print("- 'search <query>': Search command history")
    print("\nExample commands:")
    print("1. 'rename all images in C:/images to numbers from 0 to n'")
    print("2. 'создай новый файл hello.txt с текстом Привет мир'")
    print("3. 'покажи все файлы в текущей директории'")
    print("4. 'explain how python list comprehension works'")
    
    while True:
        try:
            user_input = input("\nEnter your command: ").strip()
            
            if user_input.lower() == 'exit':
                executor.speak_text("Goodbye!")
                print("Goodbye! 👋")
                break
                
            if not user_input:
                continue

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
                    executor.speak_text(error_msg, is_execution_output=True)
                    executor.history.add_command(user_input, response["code"], False, error_msg)
            elif not response.get("explanation"):
                error_msg = "Failed to generate response"
                print(f"❌ {error_msg}")
                executor.speak_text(error_msg, is_execution_output=True)
                executor.history.add_command(user_input, "", False, error_msg)
                
        except KeyboardInterrupt:
            executor.speak_text("Goodbye!")
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error: {error_msg}")
            executor.speak_text(f"Error: {error_msg}", is_execution_output=True)
            executor.history.add_command(user_input, "", False, error_msg)

if __name__ == "__main__":
    main() 