from flask import Flask, request, jsonify
from interpreter import interpreter

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message')
    model = data.get('model')

    interp = interpreter.Interpreter()
    if model:
        interp.llm.model = model

    response = interp.chat(message)
    return jsonify({'response': response})

def start_server(interpreter_instance, port=8000):
    global interp
    interp = interpreter_instance
    app.run(host='0.0.0.0', port=port)

if __name__ == '__main__':
    start_server(interpreter.Interpreter())