import argparse
from interpreter import interpreter

def cli():
    parser = argparse.ArgumentParser(description="Enhanced Interpreter CLI")
    parser.add_argument("--model", help="Specify the LLM model to use")
    parser.add_argument("--message", help="Initial message to send to the interpreter")
    args = parser.parse_args()

    interp = interpreter.Interpreter()

    if args.model:
        interp.llm.model = args.model

    if args.message:
        response = interp.chat(args.message)
        print(f"AI: {response}")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        response = interp.chat(user_input)
        print(f"AI: {response}")

if __name__ == "__main__":
    cli()