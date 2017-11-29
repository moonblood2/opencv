import numpy as np

class MyClass:
    """A simple example class"""
    i = 12345
    def __init__(self,name=''):
        self.name = name
        
    def f(self):
        if self.name == '':
            return 'hello world'
        else:
            print('hello ', self.name)




x= MyClass()
