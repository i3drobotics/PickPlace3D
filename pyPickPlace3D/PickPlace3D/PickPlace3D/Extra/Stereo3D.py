from BaslerCapture import BaslerCapture
import numpy as np
import cv2
import cv2.aruco as aruco

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

class Stereo3D():
    def __init__(self, left_cal_file, right_cal_file):
        self.calLoaded, self.m_l, self.m_r, self.d_l, self.d_r, self.r_r, self.r_l, self.p_r, self.p_l = self.get_cal_from_file(left_cal_file,right_cal_file)
        self.calc_q()
        self.ARUCO_PARAMETERS = aruco.DetectorParameters_create()
        self.ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_6X6_1000)

    def get_cal_from_file(self, left_cal_file, right_cal_file):
        res = False
        fs_l = cv2.FileStorage(left_cal_file, cv2.FILE_STORAGE_READ)
        fs_r = cv2.FileStorage(right_cal_file, cv2.FILE_STORAGE_READ)
        if fs_l.isOpened() and fs_r.isOpened():
            m_l = fs_l.getNode("camera_matrix").mat()
            m_r = fs_r.getNode("camera_matrix").mat()
            d_l = fs_l.getNode("distortion_coefficients").mat()
            d_r = fs_r.getNode("distortion_coefficients").mat()
            r_l = fs_l.getNode("rectification_matrix").mat()
            r_r = fs_r.getNode("rectification_matrix").mat()
            p_l = fs_l.getNode("projection_matrix").mat()
            p_r = fs_r.getNode("projection_matrix").mat()
            res = True
        else:
            print("Failed to open calibration files")
        fs_l.release()
        fs_r.release()
        return res, m_l, m_r, d_l, d_r, r_r, r_l, p_r, p_l

    def isCalLoaded(self):
        return self.calLoaded

    def disparity_to_depth(self,disparity):
        return self.T * self.fx / disparity

    def disparity_to_depth2(self,x,y,d):
        v = np.zeros(shape=(4,1))
        v[0,0] = x
        v[1,0] = y
        v[2,0] = d
        v[3,0] = 1

        depth = self.Q * v
        print(depth)

        return depth

    def detect_aruco(self,image,marker_length):
        rvecs, tvecs = None, None
        cXs, cYs = [], []
        corners, ids, rejectedImgPoints = aruco.detectMarkers(image, self.ARUCO_DICT, parameters=self.ARUCO_PARAMETERS)
        if (len(corners) > 0):
            rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(corners, marker_length, self.m_l, self.d_l)
            for i, r, t, c in zip(ids, rvecs, tvecs, corners):
                # compute the center of the contour
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cXs.append(cX)
                cYs.append(cY)
                print("Marker {} - X: {}, Y: {} R: {}, T: {}".format(i,cX,cY,r,t))
            
        else:
            print("No markers detected")
        return cXs, cYs

    def calc_q(self):
        q = np.zeros(shape=(4,4))
        self.cx = self.p_l[0,2]
        self.cxr = self.p_r[0,2]
        self.cy = self.p_l[1,2]
        self.fx = self.m_l[0,0]
        self.fy = self.m_l[1,1]

        p14 = self.p_r[0,3]
        self.T = -p14 / self.fx

        q33 = -(self.cx - self.cxr)/self.T

        # calucalte Q values
        q[0,0] = 1.0
        q[0,3] = -self.cx
        q[1,1] = 1.0
        q[1,3] = -self.cy #501
        q[2,3] = self.fx
        q[3,2] = 1.0 / self.T
        q[3,3] = q33

        self.Q = q
        print(self.Q)
        return q

    def rectify(self,left_image, right_image):
        mapL1, mapL2 = cv2.initUndistortRectifyMap(self.m_l, self.d_l, self.r_l, self.p_l, left_image.shape[::-1], cv2.CV_32FC1)
        mapR1, mapR2 = cv2.initUndistortRectifyMap(self.m_r, self.d_r, self.r_r, self.p_r, right_image.shape[::-1], cv2.CV_32FC1)
        rect_image_l = cv2.remap(left_image, mapL1, mapL2, cv2.INTER_LINEAR)
        rect_image_r = cv2.remap(right_image, mapR1, mapR2, cv2.INTER_LINEAR)
        return rect_image_l, rect_image_r

    def gen3D(self,left_image, right_image):
        self.min_disp = -210
        self.num_disp = 16*20

        matcher = cv2.StereoBM_create()
        matcher.setBlockSize(21)
        matcher.setMinDisparity(self.min_disp)
        matcher.setNumDisparities(self.num_disp)
        matcher.setUniquenessRatio(15)
        matcher.setTextureThreshold(10)
        matcher.setSpeckleRange(500)
        matcher.setSpeckleWindowSize(0)
        disparity = matcher.compute(left_image,right_image).astype(np.float32) / 16.0

        #disparity = (disparity-min_disp)/num_disp

        return disparity

    def genDepth(self,disparity):
        depth = cv2.reprojectImageTo3D(disparity, self.Q)
        return depth

    def write_ply(self,fn, verts, colors):
        verts = verts.reshape(-1, 3)
        colors = colors.reshape(-1, 3)

        verts = np.hstack([verts, colors])
        
        with open(fn, 'wb') as f:
            f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
            np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

    def scale_disparity(self,disparity):
        minV, maxV,_,_ = cv2.minMaxLoc(disparity)
        return cv2.convertScaleAbs(disparity, alpha=255.0/(maxV - minV), beta=-minV * 255.0/(maxV - minV))

if __name__ == '__main__':
    left_camera_serial = "22864912"
    right_camera_serial = "22864917"
    left_cal_file = "Data/left.yaml"
    right_cal_file = "Data/right.yaml"
    binning = 2

    cam_l = BaslerCapture()
    cam_r = BaslerCapture()

    print("Loading calibration...")
    stereo3D = Stereo3D(left_cal_file, right_cal_file)
    if (stereo3D.isCalLoaded()):
        print("Connecting to cameras...")
        ret_l = cam_l.connect(serial=left_camera_serial, binning=binning, inter_packet_delay=0)
        ret_r = cam_r.connect(serial=right_camera_serial, binning=binning, inter_packet_delay=0)

        if (ret_l and ret_r):
            print("Cameras connected.")
            connected = True
            while(connected):
                print("Capturing frame...")
                res_img_l,image_l = cam_l.get_image()
                res_img_r,image_r = cam_r.get_image()

                if (res_img_l and res_img_r):
                    print("Rectifying images...")
                    rect_image_l, rect_image_r = stereo3D.rectify(image_l, image_r)
                    rect_image_joint = np.concatenate((rect_image_l,rect_image_r), axis=1)
                    rect_image_joint_resized = cv2.resize(rect_image_joint, (1280, 480), interpolation = cv2.INTER_AREA)

                    cv2.imshow("camera", rect_image_joint_resized)

                    disp = stereo3D.gen3D(rect_image_l, rect_image_r)
                    disp_resized = stereo3D.scale_disparity(cv2.resize(disp, (640, 480), interpolation = cv2.INTER_AREA))

                    cv2.imshow("3D", disp_resized)

                k = cv2.waitKey(1)          
                if k == ord('q'): #exit if 'q' key pressed
                    connected = False