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
import functools
import multiprocessing
import importlib
import time
import logging
from functools import wraps
import json
import asyncio
from datetime import datetime
from interpreter.core.utils.truncate_output import truncate_output
from interpreter.terminal_interface.utils.display_markdown_message import display_markdown_message
from interpreter.terminal_interface.utils.count_tokens import count_messages_tokens
from interpreter.terminal_interface.utils.export_to_markdown import export_to_markdown
from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from collections import deque
import uvicorn

console = Console()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def error_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            console.print(f"An error occurred: {str(e)}", style="bold red")
    return wrapper

@functools.lru_cache(maxsize=100)
def cached_llm_completion(model, messages):
    return litellm.completion(
        model=model,
        messages=messages,
        stream=True
    )

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
        functions = r'\b([A-Za-z0-9_]+(?=\()))'
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
        self.add_edit_menu_actions(edit_menu)
        self.add_view_menu_actions(view_menu)
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

        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def add_edit_menu_actions(self, edit_menu):
        undo_action = QAction('Undo', self)
        undo_action.triggered.connect(self.code_editor.undo)
        edit_menu.addAction(undo_action)

        redo_action = QAction('Redo', self)
        redo_action.triggered.connect(self.code_editor.redo)
        edit_menu.addAction(redo_action)

        cut_action = QAction('Cut', self)
        cut_action.triggered.connect(self.code_editor.cut)
        edit_menu.addAction(cut_action)

        copy_action = QAction('Copy', self)
        copy_action.triggered.connect(self.code_editor.copy)
        edit_menu.addAction(copy_action)

        paste_action = QAction('Paste', self)
        paste_action.triggered.connect(self.code_editor.paste)
        edit_menu.addAction(paste_action)

    def add_view_menu_actions(self, view_menu):
        toggle_file_explorer_action = QAction('Toggle File Explorer', self)
        toggle_file_explorer_action.triggered.connect(self.toggle_file_explorer)
        view_menu.addAction(toggle_file_explorer_action)

        toggle_console_action = QAction('Toggle Console', self)
        toggle_console_action.triggered.connect(self.toggle_console)
        view_menu.addAction(toggle_console_action)

    def add_tools_menu_actions(self, tools_menu):
        voice_command_action = QAction('Voice Command', self)
        voice_command_action.triggered.connect(self.voice_command)
        tools_menu.addAction(voice_command_action)

        image_analysis_action = QAction('Image Analysis', self)
        image_analysis_action.triggered.connect(self.image_analysis)
        tools_menu.addAction(image_analysis_action)

        reload_plugins_action = QAction('Reload Plugins', self)
        reload_plugins_action.triggered.connect(self.reload_plugins)
        tools_menu.addAction(reload_plugins_action)

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
        
        for response in self.interpreter.chat(message, display=False, stream=True):
            if response['type'] == 'message':
                self.chat_display.append(f"AI: {response['content']}")
            elif response['type'] == 'code':
                self.code_editor.appendPlainText(response['content'])
            elif response['type'] == 'console':
                if response['format'] == 'output':
                    self.console_output.append(response['content'])
                elif response['format'] == 'active_line':
                    self.highlight_active_line(int(response['content']))

    def highlight_active_line(self, line_number):
        cursor = self.code_editor.textCursor()
        cursor.movePosition(QTextCursor.Start)
        cursor.movePosition(QTextCursor.Down, QTextCursor.MoveAnchor, line_number - 1)
        cursor.select(QTextCursor.LineUnderCursor)
        self.code_editor.setTextCursor(cursor)
        self.code_editor.setFocus()

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
        self.interpreter.llm.model = model
        self.chat_display.append(f"Model updated to: {model}")

    def set_custom_model(self):
        custom_model = self.custom_model_input.text()
        if custom_model:
            self.interpreter.llm.model = custom_model
            self.model_dropdown.addItem(custom_model)
            self.model_dropdown.setCurrentText(custom_model)
            self.chat_display.append(f"Custom model set: {custom_model}")
        else:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid model name.")

    def save_conversation(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save Conversation", "", "JSON Files (*.json)")
        if filename:
            with open(filename, 'w') as f:
                json.dump(self.interpreter.messages, f)

    def load_conversation(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Conversation", "", "JSON Files (*.json)")
        if filename:
            with open(filename, 'r') as f:
                self.interpreter.messages = json.load(f)
            self.update_chat_display()

    def update_chat_display(self):
        self.chat_display.clear()
        for message in self.interpreter.messages:
            if message['role'] == 'user':
                self.chat_display.append(f"You: {message['content']}")
            elif message['role'] == 'assistant':
                self.chat_display.append(f"AI: {message['content']}")

    def reset_conversation(self):
        self.interpreter.reset()
        self.chat_display.clear()
        self.code_editor.clear()
        self.console_output.clear()

    def toggle_auto_run(self):
        self.interpreter.auto_run = not self.interpreter.auto_run
        status = "enabled" if self.interpreter.auto_run else "disabled"
        self.chat_display.append(f"Auto-run {status}")
    def toggle_console(self):
        console_index = self.tab_widget.indexOf(self.tab_widget.findChild(QWidget, "Console"))
        if console_index != -1:
            self.tab_widget.removeTab(console_index)
        else:
            self.setup_console_tab()

    def reload_plugins(self):
        self.interpreter.reload_plugins()
        QMessageBox.information(self, "Plugins Reloaded", "All plugins have been reloaded.")

    def update_chat_display(self):
        self.chat_display.clear()
        for message in self.interpreter.conversation_history:
            role = message['role']
            content = message['content']
            self.chat_display.append(f"{role.capitalize()}: {content}")

class EnhancedInterpreter:
    def __init__(self, config_path: str = "config.yaml"):
        # Add new attributes
        self.offline = False
        self.auto_run = False
        self.safe_mode = "off"
        self.shrink_images = True
        self.loop = False
        self.loop_message = """Proceed. You CAN run code on my machine. If the entire task I asked for is done, say exactly 'The task is done.' If you need some specific information (like username or password) say EXACTLY 'Please provide more information.' If it's impossible, say 'The task is impossible.' (If I haven't provided a task, say exactly 'Let me know what you'd like to do next.') Otherwise keep going."""
        self.loop_breakers = [
            "The task is done.",
            "The task is impossible.",
            "Let me know what you'd like to do next.",
            "Please provide more information.",
        ]
        self.disable_telemetry = False
        self.speak_messages = False
        self.custom_instructions = ""
        self.user_message_template = "{content}"
        self.always_apply_user_message_template = False
        self.code_output_template = "Code output: {content}\n\nWhat does this output mean / what's next (if anything, or are we done)?"
        self.empty_code_output_template = "The code above was executed on my machine. It produced no text output. what's next (if anything, or are we done)?"
        self.code_output_sender = "user"
        self.sync_computer = False
        self.import_computer_api = False
        self.skills_path = None
        self.import_skills = False
        self.multi_line = True
        self.contribute_conversation = False
        self.plain_text_display = False

        self.conversation_history = []
        self.conversation_filename = None
        self.conversation_history_path = os.path.join(os.path.expanduser("~"), ".open-interpreter", "conversations")
        self.max_output = 2800
        self.system_message = "You are an AI assistant that can execute code on the user's machine."
        self.llm = Llm(self)
        self.computer = Computer(self)
        self.plugin_manager = PluginManager(os.path.join(os.path.dirname(__file__), "plugins"))
        self.plugin_manager.load_plugins()

        with open(config_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)

        self.available_models = {
            "gpt-4": "OpenAI GPT-4",
            "gpt-3.5-turbo": "OpenAI GPT-3.5 Turbo",
            "claude-2": "Anthropic Claude 2",
            "palm-2": "Google PaLM 2",
            "llama-2": "Meta LLaMA 2",
            "code-llama": "Meta Code LLaMA",
            "mistral-7b": "Mistral AI 7B",
            "openchat": "OpenChat",
            "wizardcoder": "WizardCoder",
            "stable-code": "Stability AI Stable Code",
            "custom": "Custom Model"
        }
        self.model = self.config.get('model', 'gpt-4')

        # Existing attributes...

        self.responding = False
        self.last_messages_count = 0
        self.output_queue = asyncio.Queue()
        self.unsent_messages = deque()

    async def respond(self, run_code=None):
        try:
            for chunk in self._respond_and_store():
                await self.output_queue.put(chunk)
            await self.output_queue.put(complete_message)
        except Exception as e:
            error = traceback.format_exc() + "\n" + str(e)
            error_message = {
                "role": "server",
                "type": "error",
                "content": error,
            }
            await self.output_queue.put(error_message)
            await self.output_queue.put(complete_message)
            print("\n\n--- SENT ERROR: ---\n\n")
            print(error)
            print("\n\n--- (ERROR ABOVE WAS SENT) ---\n\n")

    async def output(self):
        return await self.output_queue.get()

    def input(self, message):
        self.conversation_history.append({"role": "user", "content": message})
        self.last_messages_count = len(self.conversation_history)

    # Add new methods
    def profile(self, filename_or_url):
        return profile(self, filename_or_url)

    def reset_profile(self):
        reset_profile("default.yaml")

    def migrate_user_app_directory(self):
        migrate_user_app_directory()

    def write_key_to_profile(self, key, value):
        write_key_to_profile(key, value)

    # Update existing methods
    def chat(self, message=None, display=True, stream=False):
        if stream:
            return self._streaming_chat(message, display)
        
        for _ in self._streaming_chat(message, display):
            pass
        
        return self.conversation_history[self.last_messages_count:]

    def _streaming_chat(self, message=None, display=True):
        if display:
            yield from self._terminal_interface(message)
            return

        if message:
            self.conversation_history.append({"role": "user", "content": message})
            self.last_messages_count = len(self.conversation_history)

        yield from self._respond_and_store()

    def _respond_and_store(self):
        try:
            for chunk in self.llm.completions(messages=self.conversation_history):
                yield chunk
                if not self._is_ephemeral(chunk):
                    if chunk["type"] == "console" and chunk["format"] == "output":
                        self.conversation_history[-1]["content"] = truncate_output(
                            self.conversation_history[-1]["content"],
                            self.max_output
                        )
                    else:
                        self.conversation_history.append(chunk)
        except Exception as e:
            error = traceback.format_exc() + "\n" + str(e)
            error_message = {
                "role": "server",
                "type": "error",
                "content": error,
            }
            yield error_message
            print("\n\n--- SENT ERROR: ---\n\n")
            print(error)
            print("\n\n--- (ERROR ABOVE WAS SENT) ---\n\n")

    def _terminal_interface(self, message):
        display_markdown_message(message)
        for response in self.chat(message, display=False, stream=True):
            if response["type"] == "message":
                console.print(response["content"], end="")
            elif response["type"] == "code":
                console.print(Syntax(response["content"], response["language"], theme="monokai"))
            elif response["type"] == "image":
                image_path = response["image_path"]
                with open(image_path, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode('utf-8')
                console.print(f"![Generated Image](data:image/png;base64,{image_data})")
            elif response["type"] == "error":
                console.print(response["content"], style="bold red")
        console.print("")

    # Add authentication method
    def authenticate(self, key):
        return authenticate_function(key)

    @error_handler
    def handle_magic_command(self, user_input):
        if user_input.startswith("%%"):
            code = user_input[2:].strip()
            self.computer.run("shell", code, stream=False, display=True)
            console.print("")
            return

        command = user_input[1:].split()[0]
        arguments = user_input[len(command) + 2:].strip()

        commands = {
            "help": self.handle_help,
            "verbose": self.handle_verbose,
            "reset": self.handle_reset,
            "undo": self.handle_undo,
            "tokens": self.handle_count_tokens,
            "save_message": self.handle_save_message,
            "load_message": self.handle_load_message,
            "jupyter": self.jupyter,
            "markdown": self.markdown,
        }

        if command in commands:
            commands[command](arguments)
        else:
            console.print(f"Unknown command: {command}", style="bold red")

    def handle_help(self, arguments):
        help_text = """
        Available commands:
        %help - Display this help message
        %verbose [true/false] - Toggle verbose mode
        %reset - Reset the conversation history
        %undo - Remove the last message from the conversation history
        %tokens - Display token usage information
        %save_message [filename] - Save the current conversation to a file
        %load_message [filename] - Load a conversation from a file
        %jupyter - Switch to Jupyter notebook mode
        %markdown - Display the conversation history in markdown format
        """
        console.print(help_text)

    def handle_verbose(self, arguments):
        if arguments.lower() == "true":
            self.verbose = True
            console.print("Verbose mode enabled")
        elif arguments.lower() == "false":
            self.verbose = False
            console.print("Verbose mode disabled")
        else:
            console.print("Usage: %verbose [true/false]")

    def handle_reset(self, arguments):
        self.conversation_history = []
        console.print("Conversation history has been reset")

    def handle_undo(self, arguments):
        if self.conversation_history:
            removed_message = self.conversation_history.pop()
            console.print(f"Removed message: {removed_message['role']}: {removed_message['content'][:50]}...")
        else:
            console.print("No messages to undo")

    def handle_count_tokens(self, arguments):
        total_tokens = count_messages_tokens(self.conversation_history)
        console.print(f"Total tokens used: {total_tokens}")

    def handle_save_message(self, filename):
        if not filename:
            filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.conversation_history, f)
        console.print(f"Conversation saved to {filename}")

    def handle_load_message(self, filename):
        if not filename:
            console.print("Please provide a filename to load")
            return
        
        try:
            with open(filename, 'r') as f:
                self.conversation_history = json.load(f)
            console.print(f"Conversation loaded from {filename}")
        except FileNotFoundError:
            console.print(f"File {filename} not found")

    def jupyter(self, arguments):
        console.print("Switching to Jupyter notebook mode is not implemented in this version")

    def markdown(self, arguments):
        markdown_text = export_to_markdown(self.conversation_history)
        console.print(Markdown(markdown_text))

    @error_handler
    def execute_code(self, language: str, code: str):
        return self.computer.run(language, code, stream=True, display=True)

    @error_handler
    def process_image(self, image_path: str, query: str):
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        prompt = f"Analyze this image and answer the following question: {query}"
        response = self.chat(prompt, f"data:image/jpeg;base64,{base64_image}")
        console.print(response, style="bold green")

    @error_handler
    def voice_command(self, command: str):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            console.print("Listening...")
            audio = recognizer.listen(source)
        
        try:
            text = recognizer.recognize_google(audio)
            console.print(f"You said: {text}")
            response = self.chat(text)
            console.print(response, style="bold green")
            
            engine = pyttsx3.init()
            engine.say(response)
            engine.runAndWait()
        except sr.UnknownValueError:
            console.print("Sorry, I couldn't understand that.")
        except sr.RequestError:
            console.print("Sorry, there was an error processing your request.")

    def run(self):
        console.print(Panel.fit(
            "[bold cyan]Enhanced Interpreter[/bold cyan]\n"
            "Type 'exit' to quit, or use a magic command (e.g., %help).",
            title="Welcome",
            border_style="green",
        ))
        
        while True:
            user_input = console.input("[bold yellow]> ")
            if user_input.lower() == 'exit':
                break
            elif user_input.startswith("%"):
                self.handle_magic_command(user_input)
            else:
                for response in self.chat(user_input, stream=True):
                    if response["type"] == "message":
                        console.print(response["content"], end="")
                    elif response["type"] == "code":
                        console.print(Syntax(response["content"], response["language"], theme="monokai"))
                console.print("")

        console.print("Goodbye!")

class AsyncInterpreter(EnhancedInterpreter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_queue = asyncio.Queue()

    async def respond(self, run_code=None):
        try:
            for chunk in self._respond_and_store():
                await self.output_queue.put(chunk)
        except Exception as e:
            error = traceback.format_exc() + "\n" + str(e)
            error_message = {
                "role": "server",
                "type": "error",
                "content": error,
            }
            await self.output_queue.put(error_message)
            print("\n\n--- SENT ERROR: ---\n\n")
            print(error)
            print("\n\n--- (ERROR ABOVE WAS SENT) ---\n\n")

    async def output(self):
        return await self.output_queue.get()

def create_server(interpreter):
    async_interpreter = AsyncInterpreter(interpreter)

    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/settings")
    async def settings(payload: Dict[str, Any]):
        for key, value in payload.items():
            if key == "llm" and isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    setattr(async_interpreter, sub_key, sub_value)
            else:
                setattr(async_interpreter, key, value)
        return {"status": "success"}

    @app.websocket("/")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        try:
            async def receive_input():
                while True:
                    data = await websocket.receive()
                    if isinstance(data, bytes):
                        await async_interpreter.input(data)
                    elif "text" in data:
                        await async_interpreter.input(data["text"])
                    elif data == {"type": "websocket.disconnect", "code": 1000}:
                        print("Websocket disconnected with code 1000.")
                        break

            async def send_output():
                while True:
                    output = await async_interpreter.output()
                    if isinstance(output, dict):
                        await websocket.send_text(json.dumps(output))

            await asyncio.gather(receive_input(), send_output())
        except Exception as e:
            print(f"WebSocket connection closed with exception: {e}")
            traceback.print_exc()
        finally:
            await websocket.close()

    return app

def start_server(interpreter, port=8000):
    app = create_server(interpreter)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)

# Initialize and run the interpreter
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced Interpreter")
    parser.add_argument("--server", action="store_true", help="Run as a server")
    parser.add_argument("--port", type=int, default=8000, help="Port for the server")
    args = parser.parse_args()

    if args.server:
        start_server(args.port)
    else:
        interpreter = EnhancedInterpreter()
        interpreter.run()

class Llm:
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.model = "gpt-4"
        self.api_key = os.environ.get("OPENAI_API_KEY")

    def completions(self, messages):
        return litellm.completion(
            model=self.model,
            messages=messages,
            stream=True
        )

class Computer:
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.terminal = Terminal()

    def run(self, language, code, stream=True, display=False):
        return self.terminal.run(language, code, stream, display)

class Terminal:
    def __init__(self):
        self.languages = [Python(), JavaScript(), Shell()]

    def run(self, language, code, stream=True, display=False):
        for lang in self.languages:
            if lang.name.lower() == language.lower():
                return lang.run(code, stream, display)
        raise ValueError(f"Unsupported language: {language}")

class Python:
    name = "Python"
    
    def run(self, code, stream=True, display=False):
        try:
            output = io.StringIO()
            sys.stdout = output
            exec(code)
            sys.stdout = sys.__stdout__
            result = output.getvalue()
            if display:
                console.print(result, style="bold green")
            return result
        except Exception as e:
            error = str(e)
            if display:
                console.print(f"Error: {error}", style="bold red")
            return error

class JavaScript:
    name = "JavaScript"
    
    def run(self, code, stream=True, display=False):
        try:
            result = subprocess.run(['node', '-e', code], capture_output=True, text=True, check=True)
            output = result.stdout
            if display:
                console.print(output, style="bold green")
            return output
        except subprocess.CalledProcessError as e:
            error = e.stderr
            if display:
                console.print(f"Error: {error}", style="bold red")
            return error

class Shell:
    name = "Shell"
    
    def run(self, code, stream=True, display=False):
        try:
            result = subprocess.run(code, shell=True, capture_output=True, text=True, check=True)
            output = result.stdout
            if display:
                console.print(output, style="bold green")
            return output
        except subprocess.CalledProcessError as e:
            error = e.stderr
            if display:
                console.print(f"Error: {error}", style="bold red")
            return error

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = EnhancedInterpreterGUI()
    gui.show()
    sys.exit(app.exec())

def authenticate_function(key):
    api_key = os.getenv("INTERPRETER_API_KEY", None)
    if api_key is None:
        return True
    else:
        return key == api_key

def profile(interpreter, filename_or_url):
    if filename_or_url.startswith(('http://', 'https://')):
        response = requests.get(filename_or_url)
        profile_data = yaml.safe_load(response.text)
    else:
        with open(filename_or_url, 'r') as file:
            profile_data = yaml.safe_load(file)
    
    for key, value in profile_data.items():
        setattr(interpreter, key, value)
    
    return f"Profile loaded from {filename_or_url}"

def reset_profile(filename):
    default_profile_path = os.path.join(os.path.dirname(__file__), "profiles", filename)
    user_profile_path = os.path.join(os.path.expanduser("~"), ".open-interpreter", "profiles", filename)
    
    shutil.copy2(default_profile_path, user_profile_path)
    return f"Profile reset to default: {filename}"

def migrate_user_app_directory():
    old_dir = os.path.join(os.path.expanduser("~"), ".open-interpreter")
    new_dir = os.path.join(os.path.expanduser("~"), ".interpreter")
    
    if os.path.exists(old_dir) and not os.path.exists(new_dir):
        shutil.move(old_dir, new_dir)
        return f"Migrated from {old_dir} to {new_dir}"
    elif os.path.exists(new_dir):
        return "New directory already exists, no migration needed"
    else:
        return "Old directory not found, nothing to migrate"

def write_key_to_profile(key, value):
    profile_path = os.path.join(os.path.expanduser("~"), ".interpreter", "profiles", "default.yaml")
    
    with open(profile_path, 'r') as file:
        profile_data = yaml.safe_load(file)
    
    profile_data[key] = value
    
    with open(profile_path, 'w') as file:
        yaml.dump(profile_data, file)
    
    return f"Key '{key}' written to profile with value '{value}'"