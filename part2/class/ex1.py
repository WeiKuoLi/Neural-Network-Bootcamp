# Base class
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        print(f"{self.name} makes a sound")

# Derived class inheriting from Animal
class Dog(Animal):
    def speak(self):
        print(f"{self.name} barks")

# Creating instances of both classes
animal = Animal("Generic Animal")
dog = Dog("Buddy")

# Calling methods
animal.speak()  # Outputs: Generic Animal makes a sound
dog.speak()     # Outputs: Buddy barks

