from pyfirmata2 import Arduino        
import time                           

board = Arduino('COM3')
print("starting program")
servo1 = board.get_pin('d:2:s')
servo2 = board.get_pin('d:3:s')
servo3 = board.get_pin('d:4:s')
servo4 = board.get_pin('d:5:s')
button_pin = board.get_pin('d:12:i')  # D13 as digital input

def move(servonum, angle, time):
    
    servono = "servo" + servonum
    servono.write(angle)             
    time.sleep(time)

def pos1():
    move(1, 0, 2)
    

def wait_for_button():
    print("Waiting for button on D13 to be pressed...")
    while True:
        state = button_pin.read()
        if state is True:
            print("Button pressed!")
            break
        time.sleep(0.05)  # Poll every 50 ms


def main():
    wait_for_button()
    pos1()
    
if __name__ == "__main__":
    main()
