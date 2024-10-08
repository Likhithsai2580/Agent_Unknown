import subprocess

class CodeInterpreter:
    def __init__(self):
        self.supported_languages = ['python', 'javascript', 'bash']

    def run(self, language, code):
        if language not in self.supported_languages:
            return f"Unsupported language: {language}"

        if language == 'python':
            return self.run_python(code)
        elif language == 'javascript':
            return self.run_javascript(code)
        elif language == 'bash':
            return self.run_bash(code)

    def run_python(self, code):
        try:
            result = subprocess.run(['python', '-c', code], capture_output=True, text=True, timeout=30)
            return result.stdout if result.returncode == 0 else result.stderr
        except subprocess.TimeoutExpired:
            return "Execution timed out"

    def run_javascript(self, code):
        try:
            result = subprocess.run(['node', '-e', code], capture_output=True, text=True, timeout=30)
            return result.stdout if result.returncode == 0 else result.stderr
        except subprocess.TimeoutExpired:
            return "Execution timed out"

    def run_bash(self, code):
        try:
            result = subprocess.run(['bash', '-c', code], capture_output=True, text=True, timeout=30)
            return result.stdout if result.returncode == 0 else result.stderr
        except subprocess.TimeoutExpired:
            return "Execution timed out"

    def get_language_names(self):
        return self.supported_languages