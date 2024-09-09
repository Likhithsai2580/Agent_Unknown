import subprocess

def execute(code):
    try:
        result = subprocess.run(['node', '-e', code], capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error executing JavaScript code: {e.stderr}")

def register_plugin(interpreter):
    return {
        'name': 'JavaScript',
        'execute': execute
    }