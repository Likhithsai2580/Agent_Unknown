import subprocess

class CodeInterpreter:
    def __init__(self):
        self.supported_languages = ['python', 'javascript', 'bash', 'ruby', 'java', 'go']

    def run(self, language, code):
        if language not in self.supported_languages:
            return f"Unsupported language: {language}"

        if language == 'python':
            return self.run_python(code)
        elif language == 'javascript':
            return self.run_javascript(code)
        elif language == 'bash':
            return self.run_bash(code)
        elif language == 'ruby':
            return self.run_ruby(code)
        elif language == 'java':
            return self.run_java(code)
        elif language == 'go':
            return self.run_go(code)

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

    def run_ruby(self, code):
        try:
            result = subprocess.run(['ruby', '-e', code], capture_output=True, text=True, timeout=30)
            return result.stdout if result.returncode == 0 else result.stderr
        except subprocess.TimeoutExpired:
            return "Execution timed out"

    def run_java(self, code):
        try:
            with open('Temp.java', 'w') as file:
                file.write(code)
            result = subprocess.run(['javac', 'Temp.java'], capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return result.stderr
            result = subprocess.run(['java', 'Temp'], capture_output=True, text=True, timeout=30)
            return result.stdout if result.returncode == 0 else result.stderr
        except subprocess.TimeoutExpired:
            return "Execution timed out"
        finally:
            subprocess.run(['rm', 'Temp.java', 'Temp.class'])

    def run_go(self, code):
        try:
            with open('temp.go', 'w') as file:
                file.write(code)
            result = subprocess.run(['go', 'run', 'temp.go'], capture_output=True, text=True, timeout=30)
            return result.stdout if result.returncode == 0 else result.stderr
        except subprocess.TimeoutExpired:
            return "Execution timed out"
        finally:
            subprocess.run(['rm', 'temp.go'])

    def get_language_names(self):
        return self.supported_languages
