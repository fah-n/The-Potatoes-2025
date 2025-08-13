
from pyfirmata2 import Arduino        
import time                           

# ─────────────────── 1) เชื่อมต่อบอร์ด ───────────────────
board = Arduino('COM4')
print("เริ่มการทำงาน")

# servo1 = board.get_pin('d:2:s')
# servo2 = board.get_pin('d:3:s')
servo3 = board.get_pin('d:4:s')
# servo4 = board.get_pin('d:5:s')

try:
    while True:

        # servo4.write(60)
        # servo2.write(157)
        servo3.write(37)
        # servo1.write(20)             # write(angle) → ส่งค่ามุม (0-180°)
        time.sleep(1)                      # sหน่วง 1 s ให้หมุนถึงตำแหน่ง

except KeyboardInterrupt:
    print("\n ผู้ใช้สั่งหยุดการทำงาน")

finally:
    board.exit()  # ปิดพอร์ตอย่างปลอดภัยเมื่อจบโปรแกรม