import os
import json

def write_to_file(filename, content):
    with open(filename, 'w') as f:
        f.write(content)
    print(f"Written to {filename}")

def read_from_file(filename):
    if not os.path.exists(filename):
        print(f"File {filename} not found")
        return None
    with open(filename, 'r') as f:
        return f.read()

def append_to_file(filename, content):
    with open(filename, 'a') as f:
        f.write(content)
    print(f"Appended to {filename}")

def write_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"JSON data written to {filename}")

def read_json(filename):
    if not os.path.exists(filename):
        print(f"File {filename} not found")
        return None
    with open(filename, 'r') as f:
        return json.load(f)

def file_io_demo():
    filename = "sample.txt"
    
    write_to_file(filename, "Hello, World!\nThis is a test file.\n")
    
    content = read_from_file(filename)
    print(f"Content of {filename}:")
    print(content)
    
    append_to_file(filename, "This line was appended.\n")
    
    content = read_from_file(filename)
    print(f"Content after append:")
    print(content)
    
    data = {"name": "John", "age": 30, "skills": ["Python", "ML", "DL"]}
    json_file = "data.json"
    write_json(json_file, data)
    
    json_data = read_json(json_file)
    print(f"JSON data: {json_data}")
    
    os.remove(filename)
    os.remove(json_file)
    print("Cleanup complete")

if __name__ == "__main__":
    file_io_demo()
