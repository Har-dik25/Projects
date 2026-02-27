def guess_number():
    low = 0
    high = 10
    guesses = 0

    print("Think of a number between 0 and 10.")
    print("Answer only with 'yes' or 'no'.\n")

    while low < high:
        mid = (low + high) // 2
        guesses += 1

        answer = input(f"Is your number greater than {mid}? (yes/no): ").lower()

        if answer == "yes":
            low = mid + 1
        elif answer == "no":
            high = mid
        else:
            print("Please answer with yes or no.")

    guesses += 1
    print(f"\nYour number is: {low}")
    print(f"Total guesses taken: {guesses}")


guess_number()
