import StereoCapture as StereoCapture
import cv2
import numpy as np

class PickPlace3D():
    def __init__(self,stereo_camera):
        self.stereo_camera = stereo_camera

    def connect(self):
        return self.stereo_camera.connect()

    def run(self):
        while(True):
            res,imageL,imageR = self.stereo_camera.grab()
            if (res):
                stereo_image = np.concatenate((imageL, imageR), axis=1)
                stereo_image_resized = cv2.resize(stereo_image,(1280,480))
                cv2.imshow('Stereo Image', stereo_image_resized)
                k = cv2.waitKey(1)
                if k == ord('q'):
                    break

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
    pp.run()