from pyfirmata2 import Arduino        
import time                           

board = Arduino('COM7')
print("starting program")
servo1 = board.get_pin('d:2:s')
servo2 = board.get_pin('d:3:s')
servo3 = board.get_pin('d:4:s')
servo4 = board.get_pin('d:5:s')
button_pin = board.get_pin('d:12:i')  # D13 as digital input

def move(servonum, angle):
    if servonum == 1:
        servo1.write(angle)
    elif servonum == 2:
        servo2.write(angle)       
    elif servonum == 3:
        servo3.write(angle)
    elif servonum == 4:
        servo4.write(angle)                  

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
    # wait_for_button()
    # pos1()   
    move(2, 90)
    time.sleep(0.05)  # Poll every 50 ms
  
if __name__ == "__main__":
    main()
