import re

# =========================
# TOOLS (Agent can use ONLY these)
# =========================

def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        return "Error: Division by zero"
    return a / b


TOOLS = {
    "add": add,
    "subtract": subtract,
    "multiply": multiply,
    "divide": divide
}

# =========================
# AGENT THINKING (Decision Layer)
# =========================

def think(user_input):
    """
    Agent THINKS here.
    It decides which function to call.
    It does NOT calculate.
    """

    text = user_input.lower()
    numbers = list(map(float, re.findall(r"-?\d+\.?\d*", text)))

    if len(numbers) < 2:
        return {"tool": None, "args": None}

    if "add" in text or "plus" in text or "sum" in text:
        return {"tool": "add", "args": (numbers[0], numbers[1])}

    if "subtract" in text or "minus" in text:
        return {"tool": "subtract", "args": (numbers[0], numbers[1])}

    if "multiply" in text or "times" in text:
        return {"tool": "multiply", "args": (numbers[0], numbers[1])}

    if "divide" in text or "by" in text:
        return {"tool": "divide", "args": (numbers[0], numbers[1])}

    return {"tool": None, "args": None}


# =========================
# AGENT EXECUTOR
# =========================

def agent(user_input):
    decision = think(user_input)

    if decision["tool"] is None:
        return "No action required."

    tool = TOOLS[decision["tool"]]
    a, b = decision["args"]

    result = tool(a, b)
    return f"Result: {result}"


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    print("🤖 Python Agent Started (type exit to quit)")

    while True:
        text = input("You: ")
        if text.lower() == "exit":
            break
        print("Agent:", agent(text))
