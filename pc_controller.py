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
    'requests', 'urllib'
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
        
    def generate_code(self, user_request: str) -> Optional[str]:
        """Generate Python code based on user's natural language request."""
        try:
            # Translate request to English if needed
            english_request = self.translate_to_english(user_request)
            
            # Add available modules to the prompt
            available_modules = ", ".join(sorted(ALLOWED_MODULES))
            
            # Get successful examples from history
            examples = self.history.get_successful_examples()
            
            system_content = f"""You are a helpful assistant that generates Python code. 
            Generate ONLY the Python code without any explanation or markdown formatting.
            The code should be safe and handle errors appropriately.
            Only use the following allowed modules: {available_modules}
            Include all necessary imports from the allowed modules.
            You can use requests, but do not use anything that requires API Keys.
            Do not include ```python or ``` markers.
            When user wants to update his notes, you should update this file: r"C:\\Users\\bogda\\Documents\\nFacObs\\Notes.md" in same format.
            When user wants to update his TODOs, you should update this file: r"C:\\Users\\bogda\\Documents\\nFacObs\\Todos.md" in same format, using - [].

            For any matplotlib plots:
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
                "content": f"Generate Python code for the following request: {english_request}"
            })
            
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                temperature=0.1,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Error generating code: {str(e)}")
            return None

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
            
            # Execute the code in the isolated namespace
            exec(cleaned_code, exec_globals, local_namespace)
            return True
        except Exception as e:
            logging.error(f"Code execution error: {str(e)}")
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
    
    print("🤖 PC Controller AI - Type 'exit' to quit")
    print("You can write commands in Russian or English!")
    print("Special commands:")
    print("- 'history': Show last 5 commands")
    print("- 'search <query>': Search command history")
    print("\nExample commands:")
    print("1. 'rename all images in C:/images to numbers from 0 to n'")
    print("2. 'создай новый файл hello.txt с текстом Привет мир'")
    print("3. 'покажи все файлы в текущей директории'")
    
    while True:
        try:
            user_input = input("\nEnter your command: ").strip()
            
            if user_input.lower() == 'exit':
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
                
            print("\nGenerating and executing code...")
            code = executor.generate_code(user_input)
            
            if code:
                print("\nGenerated code:")
                print("-" * 50)
                print(code)
                print("-" * 50)
                
                if executor.validate_code(code):
                    if get_user_confirmation(code):
                        print("\nExecuting code...")
                        success = executor.execute_code(code)
                        if success:
                            print("✅ Command executed successfully!")
                            executor.history.add_command(user_input, code, True)
                        else:
                            error_msg = "Error during execution"
                            print(f"❌ {error_msg}")
                            executor.history.add_command(user_input, code, False, error_msg)
                    else:
                        print("⚠️ Code execution cancelled by user")
                        executor.history.add_command(user_input, code, False, "Cancelled by user")
                else:
                    error_msg = "Code validation failed - potentially unsafe operations detected"
                    print(f"❌ {error_msg}")
                    executor.history.add_command(user_input, code, False, error_msg)
            else:
                error_msg = "Failed to generate code"
                print(f"❌ {error_msg}")
                executor.history.add_command(user_input, "", False, error_msg)
                
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error: {error_msg}")
            executor.history.add_command(user_input, "", False, error_msg)

if __name__ == "__main__":
    main() 