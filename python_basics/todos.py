tasks = []

def add_task():
    task = input("Enter a task: ")
    tasks.append(task)
    print(f"Task '{task}' added!")

def remove_task():
    task = input("Enter a task to remove: ")
    if task in tasks:
        tasks.remove(task)
        print(f"Task '{task}' removed!")
    else:
        print(f"Task '{task}' not found!")

def view_tasks():
    if tasks:
        print("Your tasks:")
        for i, task in enumerate(tasks, 1):
            print(f"{i}. {task}")
    else:
        print("No tasks!")

def todo_list_app():
    while True:
        print("\nOptions:")
        print("1. Add task")
        print("2. Remove task")
        print("3. View tasks")
        print("4. Quit")
        
        choice = input("Choose an option: ")
        
        if choice == "1":
            add_task()
        elif choice == "2":
            remove_task()
        elif choice == "3":
            view_tasks()
        elif choice == "4":
            break
        else:
            print("Invalid option. Please choose again.")

todo_list_app()