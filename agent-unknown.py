import os
import sys
import json
import asyncio
import functools
import logging
import subprocess
from datetime import datetime
from collections import deque
import re
from typing import Any, Dict, List, Optional, Union

from rich.console import Console
from rich.syntax import Syntax
from rich.progress import Progress

# Import utilities from interpreter
from interpreter.core.utils.truncate_output import truncate_output
from interpreter.terminal_interface.utils.display_markdown_message import display_markdown_message
from interpreter.terminal_interface.utils.count_tokens import count_messages_tokens
from interpreter.terminal_interface.utils.export_to_markdown import export_to_markdown
from interpreter.core.utils import *
from interpreter.core.code_interpreters import *
from interpreter.core.llm import *
from interpreter.terminal_interface import *
from plugins import *
from interpreter.core.cli import cli
from interpreter import interpreter

# PyQt6 imports
from PyQt6.QtGui import QFont, QColor, QTextCharFormat, QSyntaxHighlighter, QTextCursor, QPalette
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QTreeView, 
    QFileSystemModel, QTabWidget, QTextEdit, QLineEdit, QPushButton, QLabel, 
    QMessageBox, QSplitter, QAction, QMenu, QFileDialog, QInputDialog, QComboBox
)
from PyQt6.QtCore import Qt, QDir, QModelIndex

# Additional imports
import psutil
import shutil
import tempfile
import speech_recognition as sr
import pyttsx3
import clipboard
import requests
from bs4 import BeautifulSoup
import cv2
import numpy as np
import base64
import yaml
import litellm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Local imports
from server import start_server

# Setup logging and console
console = Console()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def error_handler(func):
    @functools.wraps(func)
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

# Syntax Highlighter
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

# Code Editor
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

# Enhanced Interpreter GUI
class EnhancedInterpreterGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.interpreter = interpreter.Interpreter()
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
        self.model_dropdown.addItems(self.interpreter.llm.available_models)
        self.model_dropdown.setCurrentText(self.interpreter.llm.model)
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

    @error_handler
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

    @error_handler
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

    @error_handler
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

    @error_handler
    def new_file(self):
        self.code_editor.clear()
        self.tab_widget.setCurrentWidget(self.code_editor)

    @error_handler
    def open_file(self, filepath=None):
        if not filepath:
            filepath, _ = QFileDialog.getOpenFileName(self, "Open File", "", "All Files (*)")
        if filepath:
            with open(filepath, 'r') as file:
                self.code_editor.setPlainText(file.read())
            self.tab_widget.setCurrentWidget(self.code_editor)

    @error_handler
    def save_file(self):
        filepath, _ = QFileDialog.getSaveFileName(self, "Save File", "", "All Files (*)")
        if filepath:
            with open(filepath, 'w') as file:
                file.write(self.code_editor.toPlainText())

    @error_handler
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

    @error_handler
    def voice_command(self):
        import speech_recognition as sr  # Ensure you have speech_recognition installed
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

    @error_handler
    def image_analysis(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if filepath:
            query = QInputDialog.getText(self, "Image Analysis", "Enter your question about the image:")[0]
            if query:
                response = self.interpreter.chat(query, filepath)
                self.chat_display.append(f"AI: {response}")

    @error_handler
    def update_model(self, model):
        self.interpreter.llm.model = model
        self.chat_display.append(f"Model updated to: {model}")

    @error_handler
    def set_custom_model(self):
        custom_model = self.custom_model_input.text()
        if custom_model:
            self.interpreter.llm.model = custom_model
            self.model_dropdown.addItem(custom_model)
            self.model_dropdown.setCurrentText(custom_model)
            self.chat_display.append(f"Custom model set: {custom_model}")
        else:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid model name.")

    @error_handler
    def save_conversation(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save Conversation", "", "JSON Files (*.json)")
        if filename:
            with open(filename, 'w') as f:
                json.dump(self.interpreter.messages, f)

    @error_handler
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

    @error_handler
    def reset_conversation(self):
        self.interpreter.reset()
        self.chat_display.clear()
        self.code_editor.clear()
        self.console_output.clear()

    @error_handler
    def toggle_auto_run(self):
        self.interpreter.auto_run = not self.interpreter.auto_run
        status = "enabled" if self.interpreter.auto_run else "disabled"
        self.chat_display.append(f"Auto-run {status}")

    @error_handler
    def toggle_console(self):
        console_index = self.tab_widget.indexOf(self.tab_widget.findChild(QWidget, "Console"))
        if console_index != -1:
            self.tab_widget.removeTab(console_index)
        else:
            self.setup_console_tab()

    @error_handler
    def reload_plugins(self):
        self.interpreter.reload_plugins()
        QMessageBox.information(self, "Plugins Reloaded", "All plugins have been reloaded.")

    # Additional methods from interpreter modules
    def run_cli(self):
        cli()

    def execute_code_block(self, code, language):
        return self.interpreter.computer.run(language, code)

    def get_system_info(self):
        return self.interpreter.computer.system_info()

    def get_active_conversation(self):
        return self.interpreter.messages

    def set_system_message(self, message):
        self.interpreter.system_message = message

    def get_token_count(self):
        return count_messages_tokens(self.interpreter.messages)

    def export_conversation(self, format='markdown'):
        if format == 'markdown':
            return export_to_markdown(self.interpreter.messages)
        elif format == 'json':
            return json.dumps(self.interpreter.messages, indent=2)
        else:
            raise ValueError("Unsupported export format")

    def save_conversation_to_file(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.interpreter.messages, f, indent=2)

    def get_available_languages(self):
        return self.interpreter.computer.get_language_names()

    def set_max_tokens(self, max_tokens):
        self.interpreter.max_tokens = max_tokens

    def set_temperature(self, temperature):
        self.interpreter.temperature = temperature

    def toggle_safe_mode(self):
        self.interpreter.safe_mode = not self.interpreter.safe_mode
        status = "enabled" if self.interpreter.safe_mode else "disabled"
        self.chat_display.append(f"Safe mode {status}")

    def get_conversation_summary(self):
        return self.interpreter.get_conversation_summary()

    def clear_conversation(self):
        self.interpreter.messages = []
        self.update_chat_display()

    def undo_last_message(self):
        if self.interpreter.messages:
            self.interpreter.messages.pop()
            self.update_chat_display()

    def get_token_usage(self):
        return self.interpreter.get_token_usage()

    def set_api_base(self, api_base):
        self.interpreter.api_base = api_base

    def set_api_key(self, api_key):
        self.interpreter.api_key = api_key

    def get_available_models(self):
        return self.interpreter.llm.available_models

    def run_async_task(self, task):
        return asyncio.run(task)

    def get_conversation_length(self):
        return len(self.interpreter.messages)

    def get_last_response(self):
        for message in reversed(self.interpreter.messages):
            if message['role'] == 'assistant':
                return message['content']
        return None

    def get_chat_history(self):
        return self.interpreter.messages

    def add_custom_function(self, function):
        self.interpreter.custom_functions.append(function)

    def remove_custom_function(self, function_name):
        self.interpreter.custom_functions = [f for f in self.interpreter.custom_functions if f.__name__ != function_name]

    def get_custom_functions(self):
        return self.interpreter.custom_functions

    def set_conversation_context(self, context):
        self.interpreter.context = context

    def get_conversation_context(self):
        return self.interpreter.context

    def set_max_conversation_length(self, max_length):
        self.interpreter.max_conversation_length = max_length

    def get_max_conversation_length(self):
        return self.interpreter.max_conversation_length

    def truncate_conversation(self):
        if len(self.interpreter.messages) > self.interpreter.max_conversation_length:
            self.interpreter.messages = self.interpreter.messages[-self.interpreter.max_conversation_length:]
            self.update_chat_display()

    def get_model_info(self):
        return self.interpreter.llm.get_model_info()

    def set_conversation_style(self, style):
        self.interpreter.conversation_style = style

    def get_conversation_style(self):
        return self.interpreter.conversation_style

    def generate_image(self, prompt):
        return self.interpreter.generate_image(prompt)

    def analyze_image(self, image_path):
        return self.interpreter.analyze_image(image_path)

    def get_system_resources(self):
        return self.interpreter.computer.get_system_resources()

    def execute_shell_command(self, command):
        return self.interpreter.computer.execute_shell_command(command)

    def get_current_working_directory(self):
        return os.getcwd()

    def change_working_directory(self, path):
        os.chdir(path)

    def list_directory_contents(self, path='.'):
        return os.listdir(path)

    def read_file_contents(self, filepath):
        with open(filepath, 'r') as f:
            return f.read()

    def write_file_contents(self, filepath, content):
        with open(filepath, 'w') as f:
            f.write(content)

    def get_environment_variables(self):
        return dict(os.environ)

    def set_environment_variable(self, key, value):
        os.environ[key] = value

    def get_python_version(self):
        return sys.version

    def get_installed_packages(self):
        return subprocess.check_output([sys.executable, '-m', 'pip', 'list']).decode('utf-8')

    def install_package(self, package_name):
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])

    def uninstall_package(self, package_name):
        subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', package_name])

    def get_gpu_info(self):
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            return [{'id': gpu.id, 'name': gpu.name, 'load': gpu.load, 'memory': gpu.memoryUtil} for gpu in gpus]
        except ImportError:
            return "GPUtil not installed. Install it to get GPU information."

    def get_cpu_info(self):
        return psutil.cpu_percent(interval=1, percpu=True)

    def get_memory_info(self):
        memory = psutil.virtual_memory()
        return {
            'total': memory.total,
            'available': memory.available,
            'percent': memory.percent,
            'used': memory.used,
            'free': memory.free
        }

    def get_disk_info(self):
        disk = psutil.disk_usage('/')
        return {
            'total': disk.total,
            'used': disk.used,
            'free': disk.free,
            'percent': disk.percent
        }

    def get_network_info(self):
        return psutil.net_io_counters()._asdict()

    def get_battery_info(self):
        battery = psutil.sensors_battery()
        if battery:
            return {
                'percent': battery.percent,
                'power_plugged': battery.power_plugged,
                'secsleft': battery.secsleft
            }
        return None

    def get_process_list(self):
        return [p.info for p in psutil.process_iter(['pid', 'name', 'status'])]

    def kill_process(self, pid):
        try:
            process = psutil.Process(pid)
            process.terminate()
            return f"Process with PID {pid} terminated."
        except psutil.NoSuchProcess:
            return f"No process found with PID {pid}."

    def get_system_uptime(self):
        return psutil.boot_time()

    def get_system_users(self):
        return [user.name for user in psutil.users()]

    def get_system_load_average(self):
        return psutil.getloadavg()

    def get_network_connections(self):
        return [conn._asdict() for conn in psutil.net_connections()]

    def get_system_temperature(self):
        try:
            temperatures = psutil.sensors_temperatures()
            return {sensor: temps for sensor, temps in temperatures.items()}
        except AttributeError:
            return "Temperature sensors not available on this system."

    def get_system_fans(self):
        try:
            fans = psutil.sensors_fans()
            return {fan: speeds for fan, speeds in fans.items()}
        except AttributeError:
            return "Fan sensors not available on this system."

    def compress_file(self, filepath, archive_name):
        import zipfile
        with zipfile.ZipFile(archive_name, 'w') as zipf:
            zipf.write(filepath, os.path.basename(filepath))
        return f"File compressed: {archive_name}"

    def decompress_file(self, archive_path, extract_path):
        import zipfile
        with zipfile.ZipFile(archive_path, 'r') as zipf:
            zipf.extractall(extract_path)
        return f"Archive extracted to: {extract_path}"

    def get_file_hash(self, filepath, algorithm='sha256'):
        import hashlib
        hash_func = getattr(hashlib, algorithm)()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()

# The rest of the code (Llm, Computer, Terminal, language classes) remains the same

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced Interpreter")
    parser.add_argument("--server", action="store_true", help="Run as a server")
    parser.add_argument("--port", type=int, default=8000, help="Port for the server")
    args = parser.parse_args()

    if args.server:
        start_server(EnhancedInterpreter(), args.port)
    else:
        interpreter = EnhancedInterpreter()
        asyncio.run(interpreter.run())
