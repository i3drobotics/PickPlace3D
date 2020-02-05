from BaslerCapture import BaslerCapture
from StereoCalib import StereoCalib
import cv2

class CalibRoutine():
    def __init__(self,binning=1.0,isStereo=True):
        self.binning = binning
        self.stCal = StereoCalib("")
        self.isStereo = isStereo

        if (self.isStereo):
            self.cam2 = BaslerCapture()
        else:
            self.cam2 = None
        self.cam = BaslerCapture()

    def connect(self,camera_serial,camera2_serial=None,camera2_delay=0):
        #Connect to camera
        if (self.isStereo):
            print("Connecting to cameras...")
            ret2 = self.cam2.connect(serial=camera2_serial,binning=self.binning,inter_packet_delay=camera2_delay)
        else:
            print("Connecting to camera...")
            ret2 = True
        ret1 = self.cam.connect(serial=camera_serial,binning=self.binning)

        if (ret1 and ret2):
            self.isConnected = True
            print("Systems connected and ready")
        else:
            self.isConnected = False
            print("Failed to connect to camera/s")

    def process_calibration_frame(self):
        #get image from camera
        resImage,image = self.cam.get_image()
        if (self.cam2 is None):
            resImage2 = True
        else:
            resImage2,image2 = self.cam2.get_image()
        
        if (resImage and resImage2): #Image received successfully
            if (self.cam2 is None):
                img_join = image
            else:
                # rectify images if calibration is loaded
                if (self.stCal.isReady):
                    img_rect,img_rect2 = self.stCal.rectify(image,image2)
                    img_join = cv2.hconcat([img_rect, img_rect2])
                else:
                    img_join = cv2.hconcat([image, image2])
        
            #resize image for displaying
            scale_percent = 30*self.binning # percent of original size
            width = int(img_join.shape[1] * scale_percent / 100)
            height = int(img_join.shape[0] * scale_percent / 100)
            img_resized = cv2.resize(img_join, (width,height), interpolation = cv2.INTER_AREA)

            cv2.imshow('Image Feed', img_resized)

            k = cv2.waitKey(1)          
            if k == ord('q'): #exit if 'q' key pressed
                self.close_all()
            elif k == ord('g'): # grab stereo image pair
                resCal,img_join = self.stCal.process_frame(image,image2)
                if (resCal):
                    self.stCal.add_frame()
            elif k == ord('r'): # reset calibration
                self.stCal.reset()
            elif k == ord('t'): # run stereo calibration
                res = self.stCal.calibrate()
                if res:
                    self.stCal.save_calibration("/home/i3dr/Desktop/cal")
            elif k == ord('l'): #load calibration from folder
                self.stCal.load_calibration("/home/i3dr/Desktop/cal")

    def run(self):
        while(self.isConnected): #loop if cameras connected
            self.process_calibration_frame()

    def close_all(self):
        print("Closing camera connection...")
        self.cam.close()
        self.isConnected = False

if __name__ == '__main__':
    calRoutine = CalibRoutine(binning=2,isStereo=True)
    calRoutine.connect("22864912","22864917",camera2_delay=0)
    calRoutine.run()