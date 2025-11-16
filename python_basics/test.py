print("Hello, World!")

# Variables and Data Types
x = 5
name = "John"
print(x)
print(name)

# Data Types
print(type(5))  # int
print(type(3.14))  # float
print(type("hello"))  # str
print(type(True))  # bool
print(type([1, 2, 3]))  # list
print(type((1, 2, 3)))  # tuple
print(type({"name": "John", "age": 30}))  # dict


x = 15
if x > 10:
    print("x is greater than 10")
else:
    print("x is less than or equal to 10")


fruits = ["apple", "banana", "mango", "cherry"]
for fruit in fruits:
    print(fruit)


i = 0
while i < 5:
    print(i)
    i += 1

def greet(name):
    print(f"Hi, {name}!")

greet("John")
greet("Mayur")


import math
print(math.pi)
print(math.sqrt(16))