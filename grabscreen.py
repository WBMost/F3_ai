# Done by Frannecklp
import time
import cv2
import numpy as np
import dxcam
import win32gui, win32ui, win32con, win32api


class screen_grabber:
    def __init__(self,window_name = 'Fallout3'):
        # init vars to exist
        self.window_name = window_name
        self.hwnd = None
        self.pos = None
        self.camera = dxcam.create()
        

        # have clock set to check if window has moved every 3 seconds
        self.time1 = time.time()

        # actually fill vars
        self.set_hwnd()
        self.get_window_info()


    def find_window(self,name):
        def winEnumHandler(hwnd, ctx):
            if win32gui.IsWindowVisible(hwnd):
                if len(win32gui.GetWindowText(hwnd)) > 0:
                    print(hex(hwnd), '"' + win32gui.GetWindowText(hwnd) + '"') # debugging visible windows
                    
                    # grabs window iterating window name
                    title = win32gui.GetWindowText(hwnd)
                    # if window contains key string, grab and store window name
                    if title.find(name) >= 0:
                        self.window_name = title
                        print(f'title found: {self.window_name}')
        win32gui.EnumWindows(winEnumHandler, None)
    
    def set_hwnd(self):
        self.find_window(self.window_name)
        self.hwnd = win32gui.FindWindow(None, self.window_name)

    # used for specific apps that have sub windows. like Notepad
    def get_inner_windows(whndl):
        def callback(hwnd,hwnds):
            if win32gui.IsWindowVisible(hwnd) and win32gui.IsWindowEnabled(hwnd):
                hwnds[win32gui.GetClassName(hwnd)] = hwnd
            return True
        hwnds = {}
        win32gui.EnumChildWindows(whndl,callback,hwnds)
        return hwnds
    
    def get_window_info(self):
        rect = win32gui.GetWindowRect(self.hwnd)
        #needs y1 to be y1+40 for bar at top
        self.pos = (rect[0]+8,rect[1]+37,rect[2]-rect[0]-9,rect[3]-rect[1]-9)
        return self.pos
    
    def grab_screen(self):
        # get refreshed window position
        self.get_window_info()
        # print(self.pos)
        cap = self.camera.grab(region=self.pos)

        return cap