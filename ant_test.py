#! /usr/bin/env python 
# -*- coding:utf-8 -*-
#
# written by Shotaro Fujimoto, August 2014. 

from Tkinter import *
import numpy as np

class ant:
    
    def __init__(self):
        root = Tk()
        root.bind('<Key>', self.move)
        self.canvas = Canvas(root, width=640, height=480)
        self.picdata = PhotoImage(file = "ant.gif")
        self.x = 320
        self.y = 200
        self.vec = (0, 1)
        self.figure = self.canvas.create_image(self.x, self.y,
                        image=self.picdata)
        self.canvas.pack()
        root.mainloop()
    
    def draw(self):
        self.canvas.delete(self.figure)
        self.picdata = PhotoImage(file="ant.gif")
        self.figure = self.canvas.create_image(self.x, self.y, image=self.picdata)
    
    def move(self, event):
        global draw
        step = 30
        if event.keysym == 'Up':      self.y -= step
        elif event.keysym == 'Down':  self.y += step
        elif event.keysym == 'Left':  self.x -= step
        elif event.keysym == 'Right': self.x += step
        else: pass
        self.draw()

if __name__ == "__main__":
    ant()


