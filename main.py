from pyfirmata2 import Arduino, util        
import time                           

board = Arduino('COM3')
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

def setuppos(): # setup at a safe known angle --> normally once initialized it moves to 0
    move(1, 90, 0.1)
    move(2, 166, 0.1)
    move(3, 30, 0.1)
    move(4, 30, 0.1)

setuppos() #calling it as early as possible
    
def pickup1(): # leftmost donut position
    move(2, 120, 1) #arm up
    move(4, 30, 0.2) #open gripper
    move(1, 180, 0.2) #turn
    move(3, 30, 0.2)
    move(2, 140, 0.3) #arm slowly down
    move(2, 166, 0.3) #arm down
    move(4, 70, 1) #close gripper
    move(2, 120, 1) #arm up
    move(1, 90, 0.2) #return to middle position
    
def pickup2(): # topleft donut position
    move(2, 120, 1) #arm up
    move(4, 30, 0.2) #open gripper
    move(1, 135, 0.2) #turn
    move(3, 30, 0.2)
    move(2, 140, 0.3) #arm slowly down
    move(2, 166, 0.3) #arm down
    move(4, 70, 1) #close gripper
    move(2, 120, 1) #arm up
    move(1, 90, 0.2) #return to middle position
    
def placedown(): # leftmost donut position

    move(2, 120, 0.2) #arm up
    move(4, 65, 0.2) #close gripper
    move(1, 90, 0.2) #return to middle position
    move(1, 45, 2) #slowly turn to rightmost position
    move(1, 3, 2) #turn to rightmost position
    move(2, 128, 0.2) #arm slowly down
    move(2, 135, 0.5) #arm slowly down
    move(2, 147, 0.5) #arm down
    move(4, 30, 1) #open gripper
    move(2, 120, 0.5) #arm up
    move(1, 90, 0.2) #return to middle position
    
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
    
    wait_for_button()
    pickup1()
    placedown()
    pickup2()
    placedown()
    # move(2, 90, 2)
    board.exit()  # ปิดพอร์ตอย่างปลอดภัยเมื่อจบโปรแกรม
    
if __name__ == "__main__":
    main()
