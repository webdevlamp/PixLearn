class Animal:
    def __init__(self, name, species):
        self.name = name
        self.species = species
    
    def speak(self):
        return f"{self.name} makes a sound"
    
    def __str__(self):
        return f"{self.name} ({self.species})"


class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name, "Dog")
        self.breed = breed
    
    def speak(self):
        return f"{self.name} barks: Woof!"


class Cat(Animal):
    def __init__(self, name, color):
        super().__init__(name, "Cat")
        self.color = color
    
    def speak(self):
        return f"{self.name} meows: Meow!"


class Student:
    def __init__(self, name, student_id):
        self.name = name
        self.student_id = student_id
        self.grades = []
    
    def add_grade(self, grade):
        self.grades.append(grade)
    
    def get_average(self):
        if not self.grades:
            return 0
        return sum(self.grades) / len(self.grades)
    
    def get_letter_grade(self):
        avg = self.get_average()
        if avg >= 90:
            return 'A'
        elif avg >= 80:
            return 'B'
        elif avg >= 70:
            return 'C'
        elif avg >= 60:
            return 'D'
        else:
            return 'F'
    
    def __str__(self):
        return f"Student: {self.name} (ID: {self.student_id})"


def oop_demo():
    print("=== Animal Inheritance Demo ===")
    animals = [
        Dog("Buddy", "Golden Retriever"),
        Cat("Whiskers", "Orange"),
        Animal("Generic", "Unknown")
    ]
    
    for animal in animals:
        print(f"{animal} says: {animal.speak()}")
    
    print("\n=== Student Grade Management ===")
    student = Student("Alice", "S001")
    student.add_grade(85)
    student.add_grade(92)
    student.add_grade(78)
    student.add_grade(95)
    
    print(f"{student}")
    print(f"Average: {student.get_average():.2f}")
    print(f"Letter Grade: {student.get_letter_grade()}")


if __name__ == "__main__":
    oop_demo()
