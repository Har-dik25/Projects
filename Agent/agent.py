import re

# -----------------------------
# TOOL (Calculator)
# -----------------------------
def calculator(expression: str):
    try:
        return eval(expression)
    except:
        return "Invalid calculation"

# -----------------------------
# AI AGENT
# -----------------------------
class SimpleAIAgent:
    def __init__(self):
        self.tools = {
            "calculator": calculator
        }

    def think(self, user_input):
        if re.search(r"[0-9+\-*/()]", user_input):
            return "calculator"
        return "chat"

    def act(self, user_input):
        tool = self.think(user_input)

        if tool == "calculator":
            # ✅ use full input
            return self.tools["calculator"](user_input)

        else:
            return "I am a basic AI agent. I can calculate for you 🙂"

# -----------------------------
# RUN LOOP
# -----------------------------
if __name__ == "__main__":
    agent = SimpleAIAgent()

    while True:
        user = input("You: ")
        if user.lower() == "exit":
            break
        print("Agent:", agent.act(user))
