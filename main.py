from pyfirmata2 import Arduino, util        
import time                           

board = Arduino('COM9')
print("starting program")

it = util.Iterator(board)
it.start()
button_pin = board.get_pin('d:13:i')
button_pin.enable_reporting()

servos = {
    1: board.get_pin('d:2:s'),
    2: board.get_pin('d:3:s'),
    3: board.get_pin('d:4:s'),
    4: board.get_pin('d:5:s'),
}

def move(servonum, angle, duration):
    servo = servos[servonum]
    start = time.time()
    while (time.time() - start) < duration:
        servo.write(angle)
        time.sleep(0.02)  # 20 ms pulse interval
        
def movespeed(servonum, current_angle, target_angle, duration):

    servo = servos[servonum]
    steps = abs(target_angle - current_angle)
    min_step_time = 0.01  # 10 ms

    if steps == 0:
        servo.write(target_angle)
        time.sleep(duration)
        return target_angle

    step = 1 if target_angle > current_angle else -1
    step_time = max(duration / steps, min_step_time)

    moved = False
    for angle in range(current_angle, target_angle + step, step):
        servo.write(angle)
        time.sleep(step_time)
        moved = True

    if not moved or steps < 2:
        servo.write(target_angle)
        time.sleep(duration)

    return target_angle


def setuppos(): # setup at a safe known angle --> normally once initialized it moves to 0
    move(1, 90, 0.1)
    move(2, 166, 0.1)
    move(3, 37, 0.1)
    move(4, 30, 0.1)

setuppos() #calling it as early as possible
    
def pickup1(): # leftmost donut position
    movespeed(2, 147, 120, 1) #arm up
    movespeed(1, 180, 90, 0.5) #return to middle position
    move(4, 30, 0.2) #open gripper
    movespeed(1, 90, 180, 1.5) #turn to leftmost
    move(3, 37, 0.2)
    movespeed(2, 120, 166, 1) #arm down
    move(4, 80, 0.5) #close gripper
    movespeed(2, 166, 120, 1) #arm up
    movespeed(1, 180, 90, 1.5) #return to middle position
    
def pickup2(): # topleft donut position
    movespeed(2, 147, 120, 1) #arm up
    movespeed(1, 0, 90, 0.5) #return to middle position
    move(4, 30, 0.2) #open gripper
    movespeed(1, 90, 135, 1.5) #turn to topleft
    move(3, 37, 0.2)
    movespeed(2, 120, 166, 1) #arm down
    move(4, 80, 1) #close gripper
    movespeed(2, 166, 120, 1) #arm up
    movespeed(1, 135, 90, 1.5) #return to middle position
    
def pickup3(): # toprightt donut position
    movespeed(2, 147, 120, 1) #arm up
    movespeed(1, 0, 90, 1.5) #return to middle position
    move(4, 30, 0.2) #open gripper
    movespeed(1, 90, 38, 1.5) #turn to topright
    move(3, 37, 0.2)
    movespeed(2, 120, 166, 1) #arm down
    move(4, 80, 1) #close gripper
    movespeed(2, 166, 120, 1) #arm up
    movespeed(1, 90, 2, 1.5) #turn to rightmost position
    move(3, 33, 0.2)
    movespeed(2, 120, 147, 1) #arm down
    move(2, 147, 1)
    move(4, 30, 1) #open gripper
    
def pickup4(): # toprightt donut position
    move(4, 82, 1)#close gripper
    movespeed(1, 0, 93, 2) #return to middle position
    move(1, 95, 0.5)
    move(4, 30, 0.2) #open gripper
    move(2,166,0.5)
    movespeed(3, 27, 110, 0.5)
    movespeed(3, 110, 27, 0.5)
    # move(4, 80, 1)#close gripper      
    # move(4, 30, 1)#close gripper

def placedown(): # leftmost donut position
    move(4, 80, 0.2)
    movespeed(1, 90, 2, 1.5) #turn to rightmost position
    move(3, 33, 0.2)
    movespeed(2, 120, 147, 1) #arm down
    move(2, 147, 1)
    move(4, 30, 1) #open gripper
    
def wait_for_button():
    print("Waiting for button on D12 to be pressed (pyfirmata2, .value)...")
    while True:
        # .value gets updated by iterator (None = no data yet)
        if button_pin.value is False:
            print("Button pressed!")
            break
        time.sleep(0.02)  # small sleep to avoid busy loop



def main():
    setuppos()
    
    try:
        wait_for_button()
        pickup1()
        placedown()
        pickup2()
        placedown()
        pickup3()
        pickup4()
        # move(2, 90, 2)
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user (Ctrl+C). Shutting down...")
    finally:
        board.exit()  # ปิดพอร์ตอย่างปลอดภัยเมื่อจบโปรแกรม
    
if __name__ == "__main__":
    main()
