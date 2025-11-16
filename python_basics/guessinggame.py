import random

def guessing_game():
    # Generate a random number between 1 and 100
    number_to_guess = random.randint(1, 100)
    attempts = 0

    while True:
        # Get user's guess
        user_guess = input("Guess a number between 1 and 100: ")

        # Validate input
        try:
            user_guess = int(user_guess)
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue

        # Check guess
        attempts += 1
        if user_guess < number_to_guess:
            print("Too low!")
        elif user_guess > number_to_guess:
            print("Too high!")
        else:
            print(f"Congratulations! You guessed the number in {attempts} attempts.")
            break

guessing_game()