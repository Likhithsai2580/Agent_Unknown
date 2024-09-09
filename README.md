# Enhanced Interpreter

Enhanced Interpreter is a powerful and versatile AI-assisted coding environment with a graphical user interface. It combines the capabilities of large language models with various system functionalities to provide an interactive and intelligent coding experience.

## Features

- AI-powered chat interface for code assistance and general queries
- Code editor with syntax highlighting
- File explorer for easy navigation and management of your project files
- Console for executing system commands
- Support for multiple AI models, including custom models
- Voice command functionality
- Image analysis capabilities
- Dark theme for comfortable coding

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Likhithsai2580/agent-unknown.git
   cd agent-unknown
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your configuration in `config.yaml` (make sure to add your API keys for the AI models you plan to use)

## Usage

Run the Enhanced agent-unknown GUI:

- Use the chat interface to interact with the AI assistant
- Write and edit code in the Code Editor tab
- Execute system commands in the Console tab
- Use the file explorer to manage your project files
- Access additional tools like voice commands and image analysis from the Tools menu

## Detailed Feature Overview

### AI-Powered Chat Interface
Interact with an AI assistant capable of answering questions, providing code suggestions, and helping with various programming tasks.

### Code Editor
A feature-rich code editor with syntax highlighting for multiple programming languages. Write, edit, and review your code with ease.

### File Explorer
Navigate your project structure, open files, and manage your codebase directly from the GUI.

### Console
Execute system commands and view their output without leaving the application.

### Multi-Model Support
Choose from a variety of AI models or even use custom models to power your coding assistant.

### Voice Commands
Use voice input to interact with the AI assistant or execute commands hands-free.

### Image Analysis
Upload images and ask the AI to analyze them, useful for tasks like UI development or data extraction from visual sources.

### Dark Theme
A comfortable dark theme to reduce eye strain during long coding sessions.

## Configuration

The `config.yaml` file allows you to customize various aspects of the Enhanced Interpreter:

- Set your default AI model
- Configure available models and their API endpoints
- Customize the system message for the AI assistant
- Set up plugin directories

## Extending Functionality

### Plugins
The Enhanced Interpreter supports plugins to extend its functionality. Place your custom plugins in the `plugins` directory (or the directory specified in your config file).

### Agents
The application uses various agents to handle specific tasks:
- Web Search Agent
- Task Execution Agent
- File System Agent
- System Info Agent
- Code Execution Agent
- Voice Assistant Agent
- Clipboard Agent

You can extend or modify these agents to customize the behavior of the Enhanced Interpreter.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- This project uses various open-source libraries and AI models. We're grateful to the developers and researchers who made these resources available.
- Special thanks to the PyQt team for providing the framework for our GUI.

## Support

If you encounter any issues or have questions, please file an issue on the GitHub repository.