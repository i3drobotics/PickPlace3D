import serial
import time
import binascii

class RobotiqControl():
    def __init__(self,port="/dev/ttyUSB0"):
        self.ser = serial.Serial(port=port, baudrate=115200, timeout=1,parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS)

    def activate(self):
        counter = 0

        while counter < 1:

            counter = counter + 1

            self.ser.write("\x09\x10\x03\xE8\x00\x03\x06\x00\x00\x00\x00\x00\x00\x73\x30")

            data_raw = self.ser.readline()

            #print(data_raw)

            data = binascii.hexlify(data_raw)

            #print "Response 1 ", data

            time.sleep(0.01)

            self.ser.write("\x09\x03\x07\xD0\x00\x01\x85\xCF")

            data_raw = self.ser.readline()

            #print(data_raw)

            data = binascii.hexlify(data_raw)

            #print "Response 2 ", data

            #time.sleep(1)

    def close(self):
        self.ser.close()

    def close_gripper(self):
        print "Close gripper"

        self.ser.write("\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00\xFF\xFF\xFF\x42\x29")

        data_raw = self.ser.readline()

        #print(data_raw)

        data = binascii.hexlify(data_raw)

        #print "Response 3 ", data

        #time.sleep(2)

    def open_gripper(self):
        print "Open gripper"

        self.ser.write("\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00\x00\xFF\xFF\x72\x19")

        data_raw = self.ser.readline()

        #print(data_raw)

        data = binascii.hexlify(data_raw)

        #print "Response 4 ", data

        #time.sleep(2)