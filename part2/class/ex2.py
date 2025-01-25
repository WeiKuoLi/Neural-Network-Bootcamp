# Class Owner
class Owner:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def talk(self):
        print(f"{self.name} says hello!")

# Class Dog
class Dog:
    def __init__(self, name, breed):
        self.name = name
        self.breed = breed

    def greet(self, other_obj):
        print(other_obj.__class__)
        print(f"{self.name} the {self.breed} greets {other_obj.name}")
        other_obj.talk()  # Calling the 'talk' method from the Owner class

# Creating instances of both classes
dog = Dog("Buddy", "Golden Retriever")
owner = Owner("Alice", 30)

# Using a Dog method that takes an Owner object as input
dog.greet(owner)  # Outputs: Buddy the Golden Retriever greets Alice
                  #          Alice says hello!

