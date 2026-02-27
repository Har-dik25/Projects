class Array:
    def __init__(self, capacity):
        self.data = [None] * capacity
        self.size = 0
        self.capacity = capacity
    
    def traverse(self):
        for i in range(self.size):
            print(f"Index {i}: {self.data[i]}")
    
    def search(self, target):
        for i in range(self.size):
            if self.data[i] == target:
                return i
        return -1
    
    def insert(self, index, value):
        if index < 0 or index > self.size:
            raise IndexError("Invalid index")
        if self.size >= self.capacity:
            raise OverflowError("Array is full")

        # Shift elements to the right
        for i in range(self.size, index, -1):
            self.data[i] = self.data[i - 1]

        self.data[index] = value
        self.size += 1
    
    def delete(self, index):
        if index < 0 or index >= self.size:
            raise IndexError("Invalid index")

        deleted_value = self.data[index]

        # Shift elements to the left
        for i in range(index, self.size - 1):
            self.data[i] = self.data[i + 1]

        self.data[self.size - 1] = None
        self.size -= 1
        return deleted_value


# Example usage
arr = Array(10)

arr.insert(0, "Hardik")
arr.insert(1, "Tarun")
arr.insert(2, "Rashmi")
arr.insert(3, "Saurav")
arr.insert(4, "Aman")

print("Array after insertions:")
arr.traverse()

print(f"Search for Rashmi: Index {arr.search('Rashmi')}")

arr.delete(1)
print("Array after deletion:")
arr.traverse()