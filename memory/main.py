import dspy

qwq = dspy.LM("ollama_chat/llama3.2", api_base="http://localhost:11434", api_key="")
dspy.configure(lm=qwq)


def main():
    print("Hello from memory!")


if __name__ == "__main__":
    main()
