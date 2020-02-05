import cv2
import numpy as np
from pypylon import pylon
import os

class BaslerCapture():
    PIXEL_FORMAT_MONO8 = "Mono8"

    def __init__(self):
        self.camera = None

    def connect(self,serial,trigger_mode=False,pixel_format=PIXEL_FORMAT_MONO8,packet_size=3000,inter_packet_delay=0,binning=None):
        # Get the transport layer factory.
        tlFactory = pylon.TlFactory.GetInstance()

        device = None

        for d in tlFactory.GetInstance().EnumerateDevices():
            if d.GetSerialNumber() == serial:
                device = d
                break
        else:
            print('Camera with {} serial number not found'.format(serial))
            return False

        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(device))

        self.camera.Open()
        if (trigger_mode):
            self.camera.GetNodeMap().GetNode("TriggerMode").SetValue("On")
        else:
            self.camera.GetNodeMap().GetNode("TriggerMode").SetValue("Off")
        #self.camera.GetStreamGrabberParams().MaxBufferSize.SetValue(1)
        #self.camera.GetStreamGrabberParams().MaxNumBuffer.SetValue(1)
        self.camera.MaxNumBuffer = 1
        self.camera.OutputQueueSize = 1
        self.camera.GetNodeMap().GetNode("ExposureAuto").SetValue("Off")
        self.camera.GetNodeMap().GetNode("ExposureTimeRaw").SetValue(15000)
        self.camera.GetNodeMap().GetNode("GainAuto").SetValue("Off")
        self.camera.GetNodeMap().GetNode("GainRaw").SetValue(0)
        self.camera.GetNodeMap().GetNode("PixelFormat").SetValue(pixel_format)
        #self.camera.GetNodeMap().GetNode("Width").SetValue(2448)
        #self.camera.GetNodeMap().GetNode("Height").SetValue(2048)
        self.camera.GetNodeMap().GetNode("GevSCPSPacketSize").SetValue(packet_size)
        self.camera.GetNodeMap().GetNode("GevSCPD").SetValue(inter_packet_delay)
        if (not binning is None):
            self.camera.GetNodeMap().GetNode("BinningHorizontalMode").SetValue("Average")
            self.camera.GetNodeMap().GetNode("BinningHorizontal").SetValue(binning)
            self.camera.GetNodeMap().GetNode("BinningVerticalMode").SetValue("Average")
            self.camera.GetNodeMap().GetNode("BinningVertical").SetValue(binning)
        self.camera.Close()
        self.camera.StartGrabbing()

        self.converter = pylon.ImageFormatConverter()
        #converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputPixelFormat = pylon.PixelType_Mono8
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        return True

    def get_image(self):
        if (self.camera is not None):
            if (self.camera.IsGrabbing()):
                grabResult = self.camera.RetrieveResult(
                    5000, pylon.TimeoutHandling_ThrowException)

                if grabResult.GrabSucceeded():
                    img = grabResult.Array
                    image = self.converter.Convert(grabResult)
                    frame = image.GetArray()
                    return True,frame
                else:
                    print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
                    return False,None
                grabResult.Release()
            else:
                return False,None
        else:
            return False,None

    def close(self):
        self.camera.StopGrabbing()
        self.camera.Close()

if __name__ == '__main__':
    bcam = BaslerCapture("22864917")
    while(True):
        res, image = bcam.get_image()
        if (res):
            cv2.imshow('Image', image)
            k = cv2.waitKey(1)
            if k == ord('q'):
                break
    bcam.close()
