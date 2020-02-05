import StereoCapture as StereoCapture
import cv2
import numpy as np
import threading

class PickPlace3D():
    def __init__(self,stereo_camera):
        """
        Initialisation function for PickPlace3D. Defines the devices used. 
        :param stereo_camera: stereo camera used for generating 3D and 2D detection
        :type StereoCapture.StereoCapture
        """
        self.stereo_camera = stereo_camera
        self.running = False
        self.thread = None

    def connect(self):
        """
        Connect to devices needed for the pick and place process.
        :returns: success
        :rtype: bool
        """
        res = self.stereo_camera.connect()
        self.connected = res

    def update(self):
        """
        Process events. Capture and process images from stereo camera. Capture key presses.
        Should be called as frequenty as possible. (is repeatedly called by 'run()' function)
        """
        res,imageL,imageR = self.stereo_camera.grab()
        if (res):
            stereo_image = np.concatenate((imageL, imageR), axis=1)
            stereo_image_resized = cv2.resize(stereo_image,(1280,480))
            cv2.imshow('Stereo Image', stereo_image_resized)
            k = cv2.waitKey(1)
            if k == ord('q'):
                self.connected = False

    def run_blocking(self):
        """
        Run pick and place program as a blocking operation (unless called from thread).
        Should use 'run()' function rather than call this function directly.
        """
        self.running = True
        while(self.connected):
            self.update()
        self.close()
        self.running = False

    def run(self,isThreaded=False):
        """
        Run pick and place program.
        :param isThreaded: False (default) = Will run until closed by user (blocking). True = Will start run in new thread and return imediatly (threaded)
        :type isThreaded: bool
        """
        if (isThreaded):
            self.thread = threading.Thread(target=self.run)
            self.thread.start()
        else:
            self.run_blocking()

    def close(self):
        print("Quitting pick place process...")
        self.stereo_camera.close()
        print("All systems closed.")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # create opencv camera (CVCapture object)
    cvcam = StereoCapture.CVCapture(0)
    # create opencv stereo camera (StereoCaptureCVDual object)
    stcvcam = StereoCapture.StereoCaptureCVDual(cvcam)
    # create generic stereo camera (StereoCapture object)
    stcam = StereoCapture.StereoCapture(stcvcam)
    # create pick and place controller (PickPlace3D object)
    pp = PickPlace3D(stcam)
    # connect to pick and place devices
    pp.connect()
    # run pick and place routine
    pp.run(isThreaded=True)
    print("Waiting for pick place to finish...")
    while(pp.running): pass
    print("Pick place process finished.")