import math

# ===== DATA STRUCTURES =====
history = []        # ARRAY
undo_stack = []     # STACK
redo_stack = []     # STACK


def calculate():
    try:
        op = input("Operator (+ - * / ^ sqrt log): ")

        # Unary operations
        if op == "sqrt":
            a = float(input("Enter number: "))
            if a < 0:
                print(" Cannot take sqrt of negative number")
                return
            res = math.sqrt(a)
            record = f"sqrt({a}) = {res}"

        elif op == "log":
            a = float(input("Enter number: "))
            if a <= 0:
                print(" Log only defined for positive numbers")
                return
            res = math.log10(a)
            record = f"log({a}) = {res}"

        # Binary operations
        else:
            a = float(input("Enter first number: "))
            b = float(input("Enter second number: "))

            if op == '+':
                res = a + b
            elif op == '-':
                res = a - b
            elif op == '*':
                res = a * b
            elif op == '/':
                if b == 0:
                    print(" Division by zero")
                    return
                res = a / b
            elif op == '^':
                res = a ** b
            else:
                print(" Invalid operator")
                return

            record = f"{a} {op} {b} = {res}"

        # ===== ARRAY + STACK USAGE =====
        history.append(record)        # ARRAY
        undo_stack.append(res)        # STACK
        redo_stack.clear()            # STACK

        print(" Result:", res)

    except:
        print(" Invalid input")


def show_history():
    if len(history) == 0:
        print("No history available")
        return

    print("\n HISTORY (Array)")
    for i in range(len(history)):
        print(i + 1, history[i])


def undo():
    if len(undo_stack) == 0:
        print("Nothing to undo")
        return

    val = undo_stack.pop()   # STACK pop
    redo_stack.append(val)   # STACK push
    history.pop()             # ARRAY pop

    print("Undone:", val)


def redo():
    if len(redo_stack) == 0:
        print("Nothing to redo")
        return

    val = redo_stack.pop()    # STACK pop
    undo_stack.append(val)    # STACK push
    history.append(f"Redo = {val}")  # ARRAY append

    print("Redone:", val)


# ===== CLI MENU =====
while True:
    print("\n===== Calculator (Array + Stack) =====")
    print("1. Calculate")
    print("2. History")
    print("3. Undo")
    print("4. Redo")
    print("5. Exit")

    choice = input("Choose: ")

    if choice == '1':
        calculate()
    elif choice == '2':
        show_history()
    elif choice == '3':
        undo()
    elif choice == '4':
        redo()
    elif choice == '5':
        print("Exiting...")
        break
    else:
        print("Invalid choice")