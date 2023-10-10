import os
import random
import time
from traceback import format_exc
import cv2
import numpy as np
import torch
from grabscreen import screen_grabber
from game_interactions import game_functions as gf
from player import Player
import keyboard

# used to locate target window with name that contains this string
TARGET_WINDOW = 'Fallout'

def dissect_frame(full_frame):
    """
    needs to be able to dissect the frame into major aspects for information

    health, compass, npcs/environment (should use full frame actually or censored frame), AP, weapon condition, ammo (if applicable)
    """
    # needs try except as the capture can fail on rare occasion
    try:
        health_frame = full_frame[960:968,68:400]
        compasss_frame = full_frame[990:1060,70:400]
        ap_frame = full_frame[960:968,1519:1850]
        environment_frame = full_frame
    except:
        health_frame = None
        compasss_frame = None
        ap_frame = None
        environment_frame = None
        pass

    return health_frame, compasss_frame, environment_frame, ap_frame

def on_pause(player:Player):
    player.paused = not player.paused
    print(f'ai paused = {player.paused}')
    gf().pause()

if __name__ == '__main__':
    # ---- MODEL ----
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    # try to load AI model
    try:
        model = torch.load(f'{os.getcwd()}/npc_detection_model')
    except Exception as e:
        print(f'Failed to load in AI model... ERROR: {e}, {format_exc()}')
        exit()

    model.to(device=device)
    model.eval()
    

    # ---- CLASSES ----
    sg = screen_grabber(TARGET_WINDOW)
    player = Player()
    
    # ---- DEBUG ----
    # stat values
    test_hp = 37
    test_ap = 37
    timer1 = time.time()

    

    # random seed for how cautious/confident (0-100) the bot feels
    # low numbers means more likely to heal early, try to sneak in areas, shoot before initiating (good or bad)
    # high number means bot is willing to live on the edge, low health doesnt matter, go in and get dirty
    confidence_factor = random.randint(0,100)
    
    # pause and up pause feature for when alt tabbed
    keyboard.add_hotkey('p',on_pause,[player])

    while True:
        # grab window "screenshot"
        full_frame = sg.grab_screen()
        
        # generate frames for specific stats
        health_frame, compass, environment, ap_frame = dissect_frame(full_frame)
        
        # if frames properly captured, run the numbers
        if health_frame is not None:
            player.health_detection(health_frame)
        if ap_frame is not None:
            player.ap_detection(ap_frame)
        if compass is not None:
            player.compass_detection(compass)
        
        # only function if paused or not
        if not player.paused:
            player.exist()

        # sharpen and improve contrast for ai object detection
        try:
            full_frame = cv2.cvtColor(full_frame, cv2.COLOR_RGB2BGR)
            clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))
            lab = cv2.cvtColor(full_frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l2 = clahe.apply(l)
            lab = cv2.merge((l2,a,b))
            full_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        except:
            pass

        # run model detection and breakdown what is being seen
        try:
            # alter image data to be able to be processed by model
            image = np.moveaxis(full_frame,-1,0)
            image = torch.tensor(image).to(device).unsqueeze(0)
            
            # process frame
            pred = model(image)

            # assess values of predictions
            pred_results = pred[0].tolist()
            see = []
            # use weights to declare what classes are actually currently visible on the screen
            if pred_results[0] >= 8:
                see.append(f'brahmin ({pred_results[0]:.4f})')
            if pred_results[1] >= 14:
                see.append(f'centaur ({pred_results[1]:.4f})')
            if pred_results[2] >= 3.5:
                see.append(f'ghoul ({pred_results[2]:.4f})')
            if pred_results[3] >= 4:
                see.append(f'human ({pred_results[3]:.4f})')
            if pred_results[4] >= 8:
                see.append(f'landscape ({pred_results[4]:.4f})')
            if pred_results[5] >= 15:
                see.append(f'mirelurk ({pred_results[5]:.4f})')
            if pred_results[6] >= 4:
                see.append(f'super mutant ({pred_results[6]:.4f})')
            # for current debugging stages just print out what is currently visible
            if see != []:
                print('I think I see: {}'.format(', '.join(see)))
        except:
            pass

        # used for debugging the AI's view
        try:
            cv2.imshow('fullframe',full_frame)
            cv2.imshow('health',health_frame)
            # cv2.imshow('compass',compass)
            # cv2.imshow('environment',environment)
            # cv2.imshow('ap',ap_frame)
            pass
        except:
            pass
        #if abs() > 5:

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break