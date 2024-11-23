# import sys

# #class Logger(object):
# class Logger(object):
    
#     #def __init__(self):
#     def __init__(self, filename):
#         self.terminal = sys.stdout
#         #self.log = open("logfile.log", "a")
#         self.log = open(filename, "a")
   
#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)  

#     def flush(self):
#         # this flush method is needed for python 3 compatibility.
#         # this handles the flush command by doing nothing.
#         # you might want to specify some extra behavior here.
#         pass
        
import sys

class Logger(object):
    def __init__(self, filename, append=False):
        self.terminal = sys.stdout
        mode = "a" if append else "w"
        self.log = open(filename, mode)
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
"""
To use the logger and append to an existing file, you can create an instance of the Logger class with the append=True argument:
logger = Logger("logfile.log", append=True)
"""