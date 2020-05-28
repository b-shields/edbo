# -*- coding: utf-8 -*-
"""
General Utilities

"""

import time

class timer:
    """
    Returns wall clock-time
    """
    
    def __init__(self, name):
        
        self.start = time.time()
        self.name = name
        
    def stop(self):
        self.end = time.time()    
        print(self.name + ': ' + str(self.end - self.start) + ' s')


