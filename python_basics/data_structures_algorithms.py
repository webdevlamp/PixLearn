def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def bubble_sort(arr):
    n = len(arr)
    arr = arr.copy()
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

def selection_sort(arr):
    arr = arr.copy()
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        return None
    
    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        return None
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)

class Queue:
    def __init__(self):
        self.items = []
    
    def enqueue(self, item):
        self.items.append(item)
    
    def dequeue(self):
        if not self.is_empty():
            return self.items.pop(0)
        return None
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)

class LinkedListNode:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, data):
        new_node = LinkedListNode(data)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
    
    def display(self):
        elements = []
        current = self.head
        while current:
            elements.append(str(current.data))
            current = current.next
        return " -> ".join(elements)

def dsa_demo():
    print("=== Binary Search ===")
    arr = [1, 3, 5, 7, 9, 11, 13, 15]
    target = 7
    result = binary_search(arr, target)
    print(f"Array: {arr}")
    print(f"Searching for {target}: Found at index {result}")
    
    print("\n=== Sorting Algorithms ===")
    unsorted = [64, 34, 25, 12, 22, 11, 90]
    print(f"Unsorted: {unsorted}")
    print(f"Bubble Sort: {bubble_sort(unsorted)}")
    print(f"Selection Sort: {selection_sort(unsorted)}")
    
    print("\n=== Stack ===")
    stack = Stack()
    stack.push(10)
    stack.push(20)
    stack.push(30)
    print(f"Stack size: {stack.size()}")
    print(f"Peek: {stack.peek()}")
    print(f"Pop: {stack.pop()}")
    print(f"Stack size after pop: {stack.size()}")
    
    print("\n=== Queue ===")
    queue = Queue()
    queue.enqueue("First")
    queue.enqueue("Second")
    queue.enqueue("Third")
    print(f"Queue size: {queue.size()}")
    print(f"Dequeue: {queue.dequeue()}")
    print(f"Queue size after dequeue: {queue.size()}")
    
    print("\n=== Linked List ===")
    ll = LinkedList()
    ll.append(1)
    ll.append(2)
    ll.append(3)
    ll.append(4)
    print(f"Linked List: {ll.display()}")

if __name__ == "__main__":
    dsa_demo()
