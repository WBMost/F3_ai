import time
import pydirectinput as pdi
from pynput.mouse import Button, Controller


class game_functions:
    def __init__(self) -> None:
        self.sneaking = False
        self.mouse = Controller()
        pass

    def heal(self):
        print('healing')
        pdi.press('8')

    def forward(self):
        pdi.keyDown('w')
    
    def stop(self):
        pdi.keyUp('w')
        pdi.keyUp('a')
        pdi.keyUp('s')
        pdi.keyUp('d')

    def open_inventory(self):
        pdi.press('tab')

    def toggle_sneak(self):
        self.sneaking = not self.sneaking
        pdi.press('ctrl')

    def attack(self):
        pdi.leftClick()
    
    def look_x(self,val):
        pdi.moveRel(val,0)

    def pause(self):
        self.stop()
        pdi.keyUp(8)

        