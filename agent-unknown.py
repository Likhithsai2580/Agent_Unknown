import sys
import os
import yaml
import litellm
import subprocess
import base64
import cv2
import numpy as np
import pyautogui
import requests
from bs4 import BeautifulSoup
import psutil
import shutil
import tempfile
import speech_recognition as sr
import pyttsx3
import clipboard
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QTextEdit, QPushButton, QLineEdit, QLabel, QFileDialog, 
                             QTabWidget, QTreeView, QMenu, QSplitter, QMessageBox, QInputDialog, QComboBox)
from PyQt6.QtGui import QAction, QFont, QPalette, QColor, QSyntaxHighlighter, QTextCharFormat
from PyQt6.QtCore import Qt, QDir, QModelIndex, QThread, pyqtSignal
from PyQt6.QtCore import QFileSystemModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import ollama
import groq
import re
import importlib.util
from rich.console import Console
from rich.syntax import Syntax
from rich.progress import Progress
from PIL import Image

console = Console()

class SyntaxHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._mapping = {}

    def add_mapping(self, pattern, format):
        self._mapping[pattern] = format

    def highlightBlock(self, text):
        for pattern, format in self._mapping.items():
            for match in pattern.finditer(text):
                start, end = match.span()
                self.setFormat(start, end - start, format)

class CodeEditor(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFont(QFont("Courier", 12))
        self.highlighter = SyntaxHighlighter(self.document())
        self.init_highlighting()

    def init_highlighting(self):
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(Qt.GlobalColor.blue)
        keyword_format.setFontWeight(QFont.Weight.Bold)
        keywords = r'\b(def|class|import|from|as|return|if|elif|else|for|while|try|except|finally)\b'
        self.highlighter.add_mapping(re.compile(keywords), keyword_format)

        function_format = QTextCharFormat()
        function_format.setForeground(QColor("#644A9B"))
        function_format.setFontWeight(QFont.Weight.Bold)
        functions = r'\b([A-Za-z0-9_]+(?=\())'
        self.highlighter.add_mapping(re.compile(functions), function_format)

        string_format = QTextCharFormat()
        string_format.setForeground(Qt.GlobalColor.darkGreen)
        strings = r'(\'.*?\'|".*?")'
        self.highlighter.add_mapping(re.compile(strings), string_format)

class EnhancedInterpreterGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.interpreter = EnhancedInterpreter()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Enhanced Interpreter')
        self.setGeometry(100, 100, 1200, 800)

        main_widget = QWidget()
        main_layout = QHBoxLayout()

        self.setup_file_explorer()
        self.setup_tabs()
        self.setup_model_selection()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.file_explorer)
        splitter.addWidget(self.tab_widget)
        splitter.setSizes([300, 900])
        main_layout.addWidget(splitter)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        self.setup_menu_bar()
        self.apply_theme()

    def setup_file_explorer(self):
        self.file_explorer = QTreeView()
        self.file_model = QFileSystemModel()
        self.file_model.setRootPath(QDir.rootPath())
        self.file_explorer.setModel(self.file_model)
        self.file_explorer.setRootIndex(self.file_model.index(QDir.currentPath()))
        self.file_explorer.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.file_explorer.customContextMenuRequested.connect(self.show_context_menu)

    def setup_tabs(self):
        self.tab_widget = QTabWidget()
        
        self.setup_chat_tab()
        self.setup_code_editor_tab()
        self.setup_console_tab()

    def setup_chat_tab(self):
        chat_widget = QWidget()
        chat_layout = QVBoxLayout()
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_input = QLineEdit()
        self.chat_input.returnPressed.connect(self.send_message)
        chat_layout.addWidget(self.chat_display)
        chat_layout.addWidget(self.chat_input)
        chat_widget.setLayout(chat_layout)
        self.tab_widget.addTab(chat_widget, "Chat")

    def setup_code_editor_tab(self):
        self.code_editor = CodeEditor()
        self.tab_widget.addTab(self.code_editor, "Code Editor")

    def setup_console_tab(self):
        console_widget = QWidget()
        console_layout = QVBoxLayout()
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_input = QLineEdit()
        self.console_input.returnPressed.connect(self.execute_console_command)
        console_layout.addWidget(self.console_output)
        console_layout.addWidget(self.console_input)
        console_widget.setLayout(console_layout)
        self.tab_widget.addTab(console_widget, "Console")

    def setup_model_selection(self):
        model_layout = QHBoxLayout()
        model_label = QLabel("LLM Model:")
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(self.interpreter.available_models.keys())
        self.model_dropdown.setCurrentText(self.interpreter.model)
        self.model_dropdown.currentTextChanged.connect(self.update_model)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_dropdown)

        self.custom_model_input = QLineEdit()
        self.custom_model_input.setPlaceholderText("Enter custom model name")
        self.custom_model_button = QPushButton("Set Custom Model")
        self.custom_model_button.clicked.connect(self.set_custom_model)
        model_layout.addWidget(self.custom_model_input)
        model_layout.addWidget(self.custom_model_button)

        self.layout().addLayout(model_layout)

    def setup_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        edit_menu = menubar.addMenu('Edit')
        view_menu = menubar.addMenu('View')
        tools_menu = menubar.addMenu('Tools')

        self.add_file_menu_actions(file_menu)
        self.add_tools_menu_actions(tools_menu)

    def add_file_menu_actions(self, file_menu):
        new_file_action = QAction('New File', self)
        new_file_action.triggered.connect(self.new_file)
        file_menu.addAction(new_file_action)

        open_file_action = QAction('Open File', self)
        open_file_action.triggered.connect(self.open_file)
        file_menu.addAction(open_file_action)

        save_file_action = QAction('Save File', self)
        save_file_action.triggered.connect(self.save_file)
        file_menu.addAction(save_file_action)

    def add_tools_menu_actions(self, tools_menu):
        voice_command_action = QAction('Voice Command', self)
        voice_command_action.triggered.connect(self.voice_command)
        tools_menu.addAction(voice_command_action)

        image_analysis_action = QAction('Image Analysis', self)
        image_analysis_action.triggered.connect(self.image_analysis)
        tools_menu.addAction(image_analysis_action)

    def apply_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        self.setPalette(palette)

    def send_message(self):
        message = self.chat_input.text()
        self.chat_display.append(f"You: {message}")
        self.chat_input.clear()
        response = self.interpreter.chat(message)
        self.chat_display.append(f"AI: {response}")

    def execute_console_command(self):
        command = self.console_input.text()
        self.console_input.clear()
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            self.console_output.append(f"$ {command}")
            self.console_output.append(result.stdout)
            if result.stderr:
                self.console_output.append(f"Error: {result.stderr}")
        except Exception as e:
            self.console_output.append(f"Error executing command: {str(e)}")

    def show_context_menu(self, position):
        index = self.file_explorer.indexAt(position)
        if not index.isValid():
            return

        menu = QMenu()
        open_action = menu.addAction("Open")
        delete_action = menu.addAction("Delete")

        action = menu.exec(self.file_explorer.viewport().mapToGlobal(position))
        if action == open_action:
            self.open_file(self.file_model.filePath(index))
        elif action == delete_action:
            self.delete_file(self.file_model.filePath(index))

    def new_file(self):
        self.code_editor.clear()
        self.tab_widget.setCurrentWidget(self.code_editor)

    def open_file(self, filepath=None):
        if not filepath:
            filepath, _ = QFileDialog.getOpenFileName(self, "Open File", "", "All Files (*)")
        if filepath:
            with open(filepath, 'r') as file:
                self.code_editor.setPlainText(file.read())
            self.tab_widget.setCurrentWidget(self.code_editor)

    def save_file(self):
        filepath, _ = QFileDialog.getSaveFileName(self, "Save File", "", "All Files (*)")
        if filepath:
            with open(filepath, 'w') as file:
                file.write(self.code_editor.toPlainText())

    def delete_file(self, filepath):
        reply = QMessageBox.question(self, 'Delete File',
                                     f"Are you sure you want to delete {filepath}?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            try:
                os.remove(filepath)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not delete file: {str(e)}")

    def voice_command(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            self.chat_display.append("Listening...")
            audio = recognizer.listen(source)
        try:
            command = recognizer.recognize_google(audio)
            self.chat_display.append(f"You said: {command}")
            response = self.interpreter.chat(command)
            self.chat_display.append(f"AI: {response}")
        except sr.UnknownValueError:
            self.chat_display.append("Sorry, I didn't understand that.")
        except sr.RequestError:
            self.chat_display.append("Sorry, there was an error processing your request.")

    def image_analysis(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if filepath:
            query = QInputDialog.getText(self, "Image Analysis", "Enter your question about the image:")[0]
            if query:
                response = self.interpreter.chat(query, filepath)
                self.chat_display.append(f"AI: {response}")

    def update_model(self, model):
        self.interpreter.set_model(model)
        self.chat_display.append(f"Model updated to: {model}")

    def set_custom_model(self):
        custom_model = self.custom_model_input.text()
        if custom_model:
            self.interpreter.set_model(custom_model)
            self.model_dropdown.addItem(custom_model)
            self.model_dropdown.setCurrentText(custom_model)
            self.chat_display.append(f"Custom model set: {custom_model}")
        else:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid model name.")

class EnhancedInterpreter:
    def __init__(self, config_path: str = "config.yaml"):
        self.load_config(config_path)
        self.conversation_history = []
        self.plugins = {}
        self.load_plugins()
        self.init_agents()
        self.model_type = "cloud"

    def load_config(self, config_path: str):
        with open(config_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)
        self.model = self.config.get('default_model', 'gpt-3.5-turbo')
        self.available_models = self.config.get('available_models', {})
        self.system_message = self.config.get('system_message', "You are an AI assistant capable of writing and executing code in various languages, analyzing images, providing precise pixel coordinates, performing various computer tasks, and interacting via voice commands. Always ask for user confirmation before executing any code that could modify the system.")

    def load_plugins(self):
        plugin_dir = self.config.get('plugin_dir', 'plugins')
        for filename in os.listdir(plugin_dir):
            if filename.endswith('.py'):
                plugin_name = filename[:-3]
                spec = importlib.util.spec_from_file_location(plugin_name, os.path.join(plugin_dir, filename))
                plugin_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(plugin_module)
                if hasattr(plugin_module, 'register_plugin'):
                    self.plugins[plugin_name] = plugin_module.register_plugin(self)

    def set_model(self, model: str):
        self.model = model
        console.print(f"Model updated to: {self.model}", style="bold green")

    def chat(self, user_input: str, image_path: str = None):
        with Progress() as progress:
            task = progress.add_task("[cyan]Processing...", total=100)
            
            if image_path:
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                self.conversation_history.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_input},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                })
            else:
                self.conversation_history.append({"role": "user", "content": user_input})
            
            progress.update(task, advance=50)
            
            try:
                response = litellm.completion(
                    model=self.model,
                    messages=[{"role": "system", "content": self.system_message}] + self.conversation_history,
                    stream=True
                )

                full_response = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        console.print(content, end='', style="green")
                        full_response += content

                self.conversation_history.append({"role": "assistant", "content": full_response})
                
                progress.update(task, advance=50)
                
                if "```python" in full_response:
                    code_blocks = full_response.split("```python")
                    for block in code_blocks[1:]:
                        code = block.split("```")[0].strip()
                        self.execute_llm_code(code)
                else:
                    self.parse_and_execute_code(full_response)
            except Exception as e:
                console.print(f"Error: {str(e)}", style="bold red")

    def os_mode(self):
        console.print("Entering OS mode. Type 'exit' to return to normal mode.", style="bold yellow")
        while True:
            command = console.input("[bold magenta]$ ")
            if command.lower() == 'exit':
                break
            try:
                result = subprocess.run(command, shell=True, capture_output=True, text=True)
                console.print(result.stdout, style="bold blue")
                if result.stderr:
                    console.print("Error:", result.stderr, style="bold red")
            except Exception as e:
                console.print(f"Error executing command: {str(e)}", style="bold red")

    def parse_and_execute_code(self, response: str):
        code_blocks = response.split("```")
        for i in range(1, len(code_blocks), 2):
            code = code_blocks[i].strip()
            lang = code.split("\n")[0].lower()
            code = "\n".join(code.split("\n")[1:])
            
            if console.input(f"Execute {lang} code? (y/n): ").lower() == 'y':
                self.execute_code(lang, code)

    def execute_code(self, lang: str, code: str):
        self.agents["code_execution"].execute(lang, code)

    def click_image(self, description: str):
        # Take a screenshot
        screenshot = pyautogui.screenshot()
        screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

        # Use the AI to identify the location of the described element
        prompt = f"Analyze this image and provide the pixel coordinates (x, y) of the center of the element described as: '{description}'. Respond with only the coordinates in the format 'x,y'."
        
        # Convert the screenshot to base64
        _, buffer = cv2.imencode('.png', screenshot)
        base64_image = base64.b64encode(buffer).decode('utf-8')

        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an AI assistant capable of analyzing images and providing precise pixel coordinates."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]}
                ]
            )

            coordinates = response.choices[0].message.content.strip()
            x, y = map(int, coordinates.split(','))

            # Click on the identified coordinates
            pyautogui.click(x, y)
            console.print(f"Clicked at coordinates: ({x}, {y})", style="bold green")
        except Exception as e:
            console.print(f"Error identifying or clicking the image: {str(e)}", style="bold red")

    def safe_input(self, prompt):
        console.print(f"Code is requesting input: {prompt}", style="bold yellow")
        return console.input("[bold magenta]Enter your response: ")

    def execute_llm_code(self, code: str):
        console.print("The LLM has suggested the following code to execute:", style="bold yellow")
        console.print(Syntax(code, "python", theme="monokai", word_wrap=True))
        console.print("\nWarning: Executing code from an AI can be dangerous. Review carefully.", style="bold red")
        if console.input("Do you want to execute this code? (y/n): ").lower() != 'y':
            console.print("Code execution cancelled.", style="bold red")
            return

        try:
            # Create a restricted environment for code execution
            restricted_globals = {
                '__builtins__': {
                    'print': console.print,
                    'input': self.safe_input,  # Use our custom input function
                    'int': int,
                    'float': float,
                    'str': str,
                    'list': list,
                    'dict': dict,
                    'len': len,
                    'range': range,
                    'pyautogui': pyautogui,
                    'cv2': cv2,
                    'np': np,
                    'Image': Image,
                    'subprocess': subprocess,
                    'os': os,
                }
            }
            
            exec(code, restricted_globals)
        except Exception as e:
            console.print(f"Error executing code: {str(e)}", style="bold red")

    def execute_agent(self, agent_name, *args):
        if agent_name in self.agents:
            self.agents[agent_name].execute(*args)
        else:
            console.print(f"Unknown agent: {agent_name}", style="bold red")

    def run(self):
        console.print(Panel.fit(
            "[bold cyan]Enhanced Interpreter[/bold cyan]\n"
            "Type 'exit' to quit, 'os' for OS mode, 'image' to process an image,\n"
            "'click' to click on screen elements, 'code' to execute code,\n"
            "'voice' for voice commands, or use an agent command.",
            title="Welcome",
            border_style="green",
        ))
        
        while True:
            user_input = console.input("[bold yellow]> ")
            if user_input.lower() == 'exit':
                break
            elif user_input.lower() == 'os':
                self.os_mode()
            elif user_input.lower().startswith('image'):
                _, image_path = user_input.split(maxsplit=1)
                image_query = console.input("[bold magenta]Enter your question about the image: ")
                self.chat(image_query, image_path)
            elif user_input.lower().startswith('click'):
                description = console.input("[bold magenta]Describe the image or icon you want to click: ")
                self.click_image(description)
            elif user_input.lower().startswith('code'):
                _, language = user_input.split(maxsplit=1)
                code = console.input(f"[bold magenta]Enter your {language} code:\n")
                self.execute_code(language, code)
            elif user_input.lower().startswith('voice'):
                self.agents["voice_assistant"].execute("listen")
            elif user_input.lower().startswith('agent'):
                _, agent_name, *args = user_input.split()
                self.execute_agent(agent_name, *args)
            else:
                self.chat(user_input)

    def init_agents(self):
        self.agents = {
            "code_execution": CodeExecutionAgent(self),
            "voice_assistant": VoiceAssistantAgent(self),
            "image_analysis": ImageAnalysisAgent(self),
            "web_scraper": WebScraperAgent(self),
            "system_monitor": SystemMonitorAgent(self),
            "file_manager": FileManagerAgent(self),
        }

class CodeExecutionAgent:
    def __init__(self, interpreter):
        self.interpreter = interpreter

    def execute(self, lang, code):
        if lang == 'python':
            try:
                exec(code)
            except Exception as e:
                console.print(f"Error executing Python code: {str(e)}", style="bold red")
        elif lang in ['javascript', 'js']:
            try:
                subprocess.run(['node', '-e', code], check=True)
            except subprocess.CalledProcessError as e:
                console.print(f"Error executing JavaScript code: {str(e)}", style="bold red")
        else:
            console.print(f"Unsupported language: {lang}", style="bold red")

class VoiceAssistantAgent:
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()

    def execute(self, command):
        if command == "listen":
            self.listen_and_respond()
        elif command == "speak":
            text = console.input("[bold magenta]Enter text to speak: ")
            self.speak(text)

    def listen_and_respond(self):
        with sr.Microphone() as source:
            console.print("Listening...", style="bold yellow")
            audio = self.recognizer.listen(source)
        try:
            text = self.recognizer.recognize_google(audio)
            console.print(f"You said: {text}", style="bold green")
            response = self.interpreter.chat(text)
            self.speak(response)
        except sr.UnknownValueError:
            console.print("Sorry, I didn't understand that.", style="bold red")
        except sr.RequestError as e:
            console.print(f"Could not request results; {str(e)}", style="bold red")

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

class ImageAnalysisAgent:
    def __init__(self, interpreter):
        self.interpreter = interpreter

    def execute(self, image_path):
        if not os.path.exists(image_path):
            console.print(f"Image not found: {image_path}", style="bold red")
            return

        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        prompt = "Analyze this image and describe its contents in detail."
        response = self.interpreter.chat(prompt, f"data:image/jpeg;base64,{base64_image}")
        console.print(response, style="bold green")

class WebScraperAgent:
    def __init__(self, interpreter):
        self.interpreter = interpreter

    def execute(self, url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text()
            console.print(f"Content scraped from {url}:", style="bold yellow")
            console.print(text[:500] + "...", style="bold green")
        except Exception as e:
            console.print(f"Error scraping {url}: {str(e)}", style="bold red")

class SystemMonitorAgent:
    def __init__(self, interpreter):
        self.interpreter = interpreter

    def execute(self, command):
        if command == "cpu":
            cpu_percent = psutil.cpu_percent(interval=1)
            console.print(f"CPU Usage: {cpu_percent}%", style="bold green")
        elif command == "memory":
            memory = psutil.virtual_memory()
            console.print(f"Memory Usage: {memory.percent}%", style="bold green")
        elif command == "disk":
            disk = psutil.disk_usage('/')
            console.print(f"Disk Usage: {disk.percent}%", style="bold green")
        else:
            console.print(f"Unknown system monitor command: {command}", style="bold red")

class FileManagerAgent:
    def __init__(self, interpreter):
        self.interpreter = interpreter

    def execute(self, command, *args):
        if command == "list":
            self.list_files(*args)
        elif command == "create":
            self.create_file(*args)
        elif command == "delete":
            self.delete_file(*args)
        elif command == "move":
            self.move_file(*args)
        else:
            console.print(f"Unknown file manager command: {command}", style="bold red")

    def list_files(self, path="."):
        try:
            files = os.listdir(path)
            for file in files:
                console.print(file, style="bold green")
        except Exception as e:
            console.print(f"Error listing files: {str(e)}", style="bold red")

    def create_file(self, filename):
        try:
            with open(filename, 'w') as f:
                pass
            console.print(f"File created: {filename}", style="bold green")
        except Exception as e:
            console.print(f"Error creating file: {str(e)}", style="bold red")

    def delete_file(self, filename):
        try:
            os.remove(filename)
            console.print(f"File deleted: {filename}", style="bold green")
        except Exception as e:
            console.print(f"Error deleting file: {str(e)}", style="bold red")

    def move_file(self, source, destination):
        try:
            shutil.move(source, destination)
            console.print(f"File moved from {source} to {destination}", style="bold green")
        except Exception as e:
            console.print(f"Error moving file: {str(e)}", style="bold red")

class EnhancedInterpreterGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.interpreter = EnhancedInterpreter()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Enhanced Interpreter')
        self.setGeometry(100, 100, 1200, 800)

        main_widget = QWidget()
        main_layout = QHBoxLayout()

        self.setup_file_explorer()
        self.setup_tabs()
        self.setup_model_selection()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.file_explorer)
        splitter.addWidget(self.tab_widget)
        splitter.setSizes([300, 900])
        main_layout.addWidget(splitter)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        self.setup_menu_bar()
        self.apply_theme()

    def setup_file_explorer(self):
        self.file_explorer = QTreeView()
        self.file_model = QFileSystemModel()
        self.file_model.setRootPath(QDir.rootPath())
        self.file_explorer.setModel(self.file_model)
        self.file_explorer.setRootIndex(self.file_model.index(QDir.currentPath()))
        self.file_explorer.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.file_explorer.customContextMenuRequested.connect(self.show_context_menu)

    def setup_tabs(self):
        self.tab_widget = QTabWidget()
        
        self.setup_chat_tab()
        self.setup_code_editor_tab()
        self.setup_console_tab()

    def setup_chat_tab(self):
        chat_widget = QWidget()
        chat_layout = QVBoxLayout()
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_input = QLineEdit()
        self.chat_input.returnPressed.connect(self.send_message)
        chat_layout.addWidget(self.chat_display)
        chat_layout.addWidget(self.chat_input)
        chat_widget.setLayout(chat_layout)
        self.tab_widget.addTab(chat_widget, "Chat")

    def setup_code_editor_tab(self):
        self.code_editor = CodeEditor()
        self.tab_widget.addTab(self.code_editor, "Code Editor")

    def setup_console_tab(self):
        console_widget = QWidget()
        console_layout = QVBoxLayout()
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_input = QLineEdit()
        self.console_input.returnPressed.connect(self.execute_console_command)
        console_layout.addWidget(self.console_output)
        console_layout.addWidget(self.console_input)
        console_widget.setLayout(console_layout)
        self.tab_widget.addTab(console_widget, "Console")

    def setup_model_selection(self):
        model_layout = QHBoxLayout()
        model_label = QLabel("LLM Model:")
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(self.interpreter.available_models.keys())
        self.model_dropdown.setCurrentText(self.interpreter.model)
        self.model_dropdown.currentTextChanged.connect(self.update_model)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_dropdown)

        self.custom_model_input = QLineEdit()
        self.custom_model_input.setPlaceholderText("Enter custom model name")
        self.custom_model_button = QPushButton("Set Custom Model")
        self.custom_model_button.clicked.connect(self.set_custom_model)
        model_layout.addWidget(self.custom_model_input)
        model_layout.addWidget(self.custom_model_button)

        self.layout().addLayout(model_layout)

    def setup_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        edit_menu = menubar.addMenu('Edit')
        view_menu = menubar.addMenu('View')
        tools_menu = menubar.addMenu('Tools')

        self.add_file_menu_actions(file_menu)
        self.add_tools_menu_actions(tools_menu)

    def add_file_menu_actions(self, file_menu):
        new_file_action = QAction('New File', self)
        new_file_action.triggered.connect(self.new_file)
        file_menu.addAction(new_file_action)

        open_file_action = QAction('Open File', self)
        open_file_action.triggered.connect(self.open_file)
        file_menu.addAction(open_file_action)

        save_file_action = QAction('Save File', self)
        save_file_action.triggered.connect(self.save_file)
        file_menu.addAction(save_file_action)

    def add_tools_menu_actions(self, tools_menu):
        voice_command_action = QAction('Voice Command', self)
        voice_command_action.triggered.connect(self.voice_command)
        tools_menu.addAction(voice_command_action)

        image_analysis_action = QAction('Image Analysis', self)
        image_analysis_action.triggered.connect(self.image_analysis)
        tools_menu.addAction(image_analysis_action)

    def apply_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        self.setPalette(palette)

    def send_message(self):
        message = self.chat_input.text()
        self.chat_display.append(f"You: {message}")
        self.chat_input.clear()
        response = self.interpreter.chat(message)
        self.chat_display.append(f"AI: {response}")

    def execute_console_command(self):
        command = self.console_input.text()
        self.console_input.clear()
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            self.console_output.append(f"$ {command}")
            self.console_output.append(result.stdout)
            if result.stderr:
                self.console_output.append(f"Error: {result.stderr}")
        except Exception as e:
            self.console_output.append(f"Error executing command: {str(e)}")

    def show_context_menu(self, position):
        index = self.file_explorer.indexAt(position)
        if not index.isValid():
            return

        menu = QMenu()
        open_action = menu.addAction("Open")
        delete_action = menu.addAction("Delete")

        action = menu.exec(self.file_explorer.viewport().mapToGlobal(position))
        if action == open_action:
            self.open_file(self.file_model.filePath(index))
        elif action == delete_action:
            self.delete_file(self.file_model.filePath(index))

    def new_file(self):
        self.code_editor.clear()
        self.tab_widget.setCurrentWidget(self.code_editor)

    def open_file(self, filepath=None):
        if not filepath:
            filepath, _ = QFileDialog.getOpenFileName(self, "Open File", "", "All Files (*)")
        if filepath:
            with open(filepath, 'r') as file:
                self.code_editor.setPlainText(file.read())
            self.tab_widget.setCurrentWidget(self.code_editor)

    def save_file(self):
        filepath, _ = QFileDialog.getSaveFileName(self, "Save File", "", "All Files (*)")
        if filepath:
            with open(filepath, 'w') as file:
                file.write(self.code_editor.toPlainText())

    def delete_file(self, filepath):
        reply = QMessageBox.question(self, 'Delete File',
                                     f"Are you sure you want to delete {filepath}?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            try:
                os.remove(filepath)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not delete file: {str(e)}")

    def voice_command(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            self.chat_display.append("Listening...")
            audio = recognizer.listen(source)
        try:
            command = recognizer.recognize_google(audio)
            self.chat_display.append(f"You said: {command}")
            response = self.interpreter.chat(command)
            self.chat_display.append(f"AI: {response}")
        except sr.UnknownValueError:
            self.chat_display.append("Sorry, I didn't understand that.")
        except sr.RequestError:
            self.chat_display.append("Sorry, there was an error processing your request.")

    def image_analysis(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if filepath:
            query = QInputDialog.getText(self, "Image Analysis", "Enter your question about the image:")[0]
            if query:
                response = self.interpreter.chat(query, filepath)
                self.chat_display.append(f"AI: {response}")

    def update_model(self, model):
        self.interpreter.set_model(model)
        self.chat_display.append(f"Model updated to: {model}")

    def set_custom_model(self):
        custom_model = self.custom_model_input.text()
        if custom_model:
            self.interpreter.set_model(custom_model)
            self.model_dropdown.addItem(custom_model)
            self.model_dropdown.setCurrentText(custom_model)
            self.chat_display.append(f"Custom model set: {custom_model}")
        else:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid model name.")

class EnhancedInterpreter:
    def __init__(self, config_path: str = "config.yaml"):
        self.load_config(config_path)
        self.conversation_history = []
        self.plugins = {}
        self.load_plugins()
        self.init_agents()
        self.model_type = "cloud"

    def load_config(self, config_path: str):
        with open(config_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)
        self.model = self.config.get('default_model', 'gpt-3.5-turbo')
        self.available_models = self.config.get('available_models', {})
        self.system_message = self.config.get('system_message', "You are an AI assistant capable of writing and executing code in various languages, analyzing images, providing precise pixel coordinates, performing various computer tasks, and interacting via voice commands. Always ask for user confirmation before executing any code that could modify the system.")

    def load_plugins(self):
        plugin_dir = self.config.get('plugin_dir', 'plugins')
        for filename in os.listdir(plugin_dir):
            if filename.endswith('.py'):
                plugin_name = filename[:-3]
                spec = importlib.util.spec_from_file_location(plugin_name, os.path.join(plugin_dir, filename))
                plugin_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(plugin_module)
                if hasattr(plugin_module, 'register_plugin'):
                    self.plugins[plugin_name] = plugin_module.register_plugin(self)

    def set_model(self, model: str):
        self.model = model
        console.print(f"Model updated to: {self.model}", style="bold green")

    def chat(self, user_input: str, image_path: str = None):
        with Progress() as progress:
            task = progress.add_task("[cyan]Processing...", total=100)
            
            if image_path:
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                self.conversation_history.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_input},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                })
            else:
                self.conversation_history.append({"role": "user", "content": user_input})
            
            progress.update(task, advance=50)
            
            try:
                response = litellm.completion(
                    model=self.model,
                    messages=[{"role": "system", "content": self.system_message}] + self.conversation_history,
                    stream=True
                )

                full_response = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        console.print(content, end='', style="green")
                        full_response += content

                self.conversation_history.append({"role": "assistant", "content": full_response})
                
                progress.update(task, advance=50)
                
                if "python" in full_response:
                    code_blocks = full_response.split("```python")
                    for block in code_blocks[1:]:
                        code = block.split("```")[0].strip()
                        self.execute_llm_code(code)
                else:
                    self.parse_and_execute_code(full_response)
            except Exception as e:
                console.print(f"Error: {str(e)}", style="bold red")

    def os_mode(self):
        console.print("Entering OS mode. Type 'exit' to return to normal mode.", style="bold yellow")
        while True:
            command = console.input("[bold magenta]$ ")
            if command.lower() == 'exit':
                break
            try:
                result = subprocess.run(command, shell=True, capture_output=True, text=True)
                console.print(result.stdout, style="bold blue")
                if result.stderr:
                    console.print("Error:", result.stderr, style="bold red")
            except Exception as e:
                console.print(f"Error executing command: {str(e)}", style="bold red")

    def parse_and_execute_code(self, response: str):
        code_blocks = response.split("```")
        for i in range(1, len(code_blocks), 2):
            code = code_blocks[i].strip()
            lang = code.split("\n")[0].lower()
            code = "\n".join(code.split("\n")[1:])
            
            if console.input(f"Execute {lang} code? (y/n): ").lower() == 'y':
                self.execute_code(lang, code)

    def execute_code(self, lang: str, code: str):
        self.agents["code_execution"].execute(lang, code)

    def click_image(self, description: str):
        # Take a screenshot
        screenshot = pyautogui.screenshot()
        screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

        # Use the AI to identify the location of the described element
        prompt = f"Analyze this image and provide the pixel coordinates (x, y) of the center of the element described as: '{description}'. Respond with only the coordinates in the format 'x,y'."
        
        # Convert the screenshot to base64
        _, buffer = cv2.imencode('.png', screenshot)
        base64_image = base64.b64encode(buffer).decode('utf-8')

        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an AI assistant capable of analyzing images and providing precise pixel coordinates."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]}
                ]
            )

            coordinates = response.choices[0].message.content.strip()
            x, y = map(int, coordinates.split(','))

            # Click on the identified coordinates
            pyautogui.click(x, y)
            console.print(f"Clicked at coordinates: ({x}, {y})", style="bold green")
        except Exception as e:
            console.print(f"Error identifying or clicking the image: {str(e)}", style="bold red")

    def safe_input(self, prompt):
        console.print(f"Code is requesting input: {prompt}", style="bold yellow")
        return console.input("[bold magenta]Enter your response: ")

    def execute_llm_code(self, code: str):
        console.print("The LLM has suggested the following code to execute:", style="bold yellow")
        console.print(Syntax(code, "python", theme="monokai", word_wrap=True))
        console.print("\nWarning: Executing code from an AI can be dangerous. Review carefully.", style="bold red")
        if console.input("Do you want to execute this code? (y/n): ").lower() != 'y':
            console.print("Code execution cancelled.", style="bold red")
            return

        try:
            # Create a restricted environment for code execution
            restricted_globals = {
                '__builtins__': {
                    'print': console.print,
                    'input': self.safe_input,  # Use our custom input function
                    'int': int,
                    'float': float,
                    'str': str,
                    'list': list,
                    'dict': dict,
                    'len': len,
                    'range': range,
                    'pyautogui': pyautogui,
                    'cv2': cv2,
                    'np': np,
                    'Image': Image,
                    'subprocess': subprocess,
                    'os': os,
                }
            }
            
            exec(code, restricted_globals)
        except Exception as e:
            console.print(f"Error executing code: {str(e)}", style="bold red")

    def execute_agent(self, agent_name, *args):
        if agent_name in self.agents:
            self.agents[agent_name].execute(*args)
        else:
            console.print(f"Unknown agent: {agent_name}", style="bold red")

    def run(self):
        console.print(Panel.fit(
            "[bold cyan]Enhanced Interpreter[/bold cyan]\n"
            "Type 'exit' to quit, 'os' for OS mode, 'image' to process an image,\n"
            "'click' to click on screen elements, 'code' to execute code,\n"
            "'voice' for voice commands, or use an agent command.",
            title="Welcome",
            border_style="green",
        ))
        
        while True:
            user_input = console.input("[bold yellow]> ")
            if user_input.lower() == 'exit':
                break
            elif user_input.lower() == 'os':
                self.os_mode()
            elif user_input.lower().startswith('image'):
                _, image_path = user_input.split(maxsplit=1)
                image_query = console.input("[bold magenta]Enter your question about the image: ")
                self.chat(image_query, image_path)
            elif user_input.lower().startswith('click'):
                description = console.input("[bold magenta]Describe the image or icon you want to click: ")
                self.click_image(description)
            elif user_input.lower().startswith('code'):
                _, language = user_input.split(maxsplit=1)
                code = console.input(f"[bold magenta]Enter your {language} code:\n")
                self.execute_code(language, code)
            elif user_input.lower().startswith('voice'):
                self.agents["voice_assistant"].execute("listen")
            elif user_input.lower().startswith('agent'):
                _, agent_name, *args = user_input.split()
                self.execute_agent(agent_name, *args)
            else:
                self.chat(user_input)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = EnhancedInterpreterGUI()
    gui.show()
    sys.exit(app.exec())