import StereoCapture
import PickPlace3D

# create opencv camera (CVCapture object)
cvcam = StereoCapture.CVCapture(0)
# create opencv stereo camera (StereoCaptureCVDual object)
stcvcam = StereoCapture.StereoCaptureCVDual(cvcam)
# create generic stereo camera (StereoCapture object)
stcam = StereoCapture.StereoCapture(stcvcam)
# create pick and place controller (PickPlace3D object)
pp = PickPlace3D.PickPlace3D(stcam)
# connect to pick and place devices
pp.connect()
# run pick and place routine
pp.run()