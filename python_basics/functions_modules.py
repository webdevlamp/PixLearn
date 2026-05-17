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

def factorial(n):
    if n < 0:
        return "Error: Negative number"
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[-1] + fib[-2])
    return fib

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def celsius_to_fahrenheit(c):
    return (c * 9/5) + 32

def fahrenheit_to_celsius(f):
    return (f - 32) * 5/9

def functions_demo():
    print("=== Math Operations ===")
    print(f"10 + 5 = {add(10, 5)}")
    print(f"10 - 5 = {subtract(10, 5)}")
    print(f"10 * 5 = {multiply(10, 5)}")
    print(f"10 / 5 = {divide(10, 5)}")
    print(f"10 / 0 = {divide(10, 0)}")
    
    print("\n=== Factorial ===")
    print(f"5! = {factorial(5)}")
    print(f"0! = {factorial(0)}")
    print(f"-1! = {factorial(-1)}")
    
    print("\n=== Fibonacci ===")
    print(f"First 10: {fibonacci(10)}")
    
    print("\n=== Prime Numbers ===")
    primes = [n for n in range(1, 30) if is_prime(n)]
    print(f"Primes under 30: {primes}")
    
    print("\n=== Temperature Conversion ===")
    print(f"100C = {celsius_to_fahrenheit(100)}F")
    print(f"212F = {fahrenheit_to_celsius(212)}C")

if __name__ == "__main__":
    functions_demo()
