class Stack:
    def __init__(self):
        self.data = []

    def push(self, value):
        self.data.append(value)

    def pop(self):
        if not self.data:
            return None
        return self.data.pop()

    def is_empty(self):
        return len(self.data) == 0


class TextEditor:
    def __init__(self):
        self.text = ""
        self.undo_stack = Stack()
        self.redo_stack = Stack()

    def type(self, new_text):
        self.undo_stack.push(self.text)   
        self.text += new_text
        self.redo_stack = Stack()         

    def delete(self, n):
        self.undo_stack.push(self.text)
        self.text = self.text[:-n]
        self.redo_stack = Stack()

    def undo(self):
        if self.undo_stack.is_empty():
            print("Nothing to undo")
            return
        self.redo_stack.push(self.text)
        self.text = self.undo_stack.pop()

    def redo(self):
        if self.redo_stack.is_empty():
            print("Nothing to redo")
            return
        self.undo_stack.push(self.text)
        self.text = self.redo_stack.pop()

    def show(self):
        print("Text:", self.text)




editor = TextEditor()

editor.type("Hello")
editor.show()

editor.type(" World")
editor.show()

editor.delete(6)
editor.show()

editor.undo()
editor.show()

editor.redo()
editor.show()