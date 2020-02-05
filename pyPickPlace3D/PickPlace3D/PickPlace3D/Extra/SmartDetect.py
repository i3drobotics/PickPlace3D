from Stereo3D import Stereo3D
from BaslerCapture import BaslerCapture
from SquareDetect import SquareDetect, DetectObject
import numpy as np
import cv2

class SmartDetect():
    def __init__(self,left_camera_serial,right_camera_serial,left_cal_file,right_cal_file,binning):
        self.connected = False
        self.binning = binning
        self.cam_l = BaslerCapture()
        self.cam_r = BaslerCapture()
        self.sqDetect = SquareDetect(binning=self.binning,isShowThreshold=True)
        print("Loading calibration...")
        self.stereo3D = Stereo3D(left_cal_file, right_cal_file)
        if (self.stereo3D.isCalLoaded()):
            #connect to cameras
            print("Connecting to cameras...")
            ret_l = self.cam_l.connect(left_camera_serial,binning=binning)
            ret_r = self.cam_r.connect(right_camera_serial,binning=binning)
            if (ret_l and ret_r):
                self.connected = True

    def click_position(self,event,x,y,flags,para):
        if event == cv2.EVENT_LBUTTONDOWN:
            bin_x = x * self.binning
            bin_y = y * self.binning
            print([(bin_x, bin_y)])

    def run(self):
        #cv2.namedWindow("left")
        #cv2.namedWindow("right")
        #cv2.namedWindow("disp")
        #cv2.setMouseCallback("left", self.click_position)
        #cv2.setMouseCallback("right", self.click_position)
        #cv2.setMouseCallback("disp", self.click_position)
        while(self.connected):
            res_img_l,image_l = self.cam_l.get_image()
            res_img_r,image_r = self.cam_r.get_image()
            if (res_img_l and res_img_r):
                rect_image_l, rect_image_r = self.stereo3D.rectify(image_l, image_r)

                #rect_image_l = np.uint8(rect_image_l)
                #rect_image_r = np.uint8(rect_image_r)

                rect_image_l_resize = cv2.resize(rect_image_l, (640, 480), interpolation = cv2.INTER_AREA)
                rect_image_r_resize = cv2.resize(rect_image_r, (640, 480), interpolation = cv2.INTER_AREA)

                #cv2.imshow("left", rect_image_l)
                #cv2.imshow("right", rect_image_r)

                rect_image_joint = np.concatenate((rect_image_l_resize,rect_image_r_resize), axis=1)
                #cv2.imshow("cameras", rect_image_joint)

                disp = self.stereo3D.gen3D(rect_image_l, rect_image_r)
                # generate 3D points from disparity
                points = self.stereo3D.genDepth(disp)
                # remove inf values
                points[points >= 1E308] = 0
                #points[points < 0] = 0

                disp_scaled = self.stereo3D.scale_disparity(disp)
                disp_resized = cv2.resize(disp_scaled, (640, 480), interpolation = cv2.INTER_AREA)
                #cv2.imshow("3D", disp_resized)
                #cv2.imshow("disp", disp_scaled)

                disp_image_joint = np.concatenate((rect_image_l_resize,disp_resized,rect_image_r_resize), axis=1)
                #cv2.imshow("Stereo 3D", disp_image_joint)

                ROI_l = [598,800,462,1500]
                ROI_r = [608,800,746,1500]
                #ROI_d = [650,800,560,1500]
                ROI_d = [598,800,462,1500]

                # scale by binning
                ROI_l[:] = [x / self.binning for x in ROI_l]
                ROI_r[:] = [x / self.binning for x in ROI_r]
                ROI_d[:] = [x / self.binning for x in ROI_d]

                # crop images
                rect_image_l_crop = rect_image_l[ROI_l[0]:ROI_l[0]+ROI_l[1],ROI_l[2]:ROI_l[2]+ROI_l[3]]
                rect_image_r_crop = rect_image_r[ROI_r[0]:ROI_r[0]+ROI_r[1],ROI_r[2]:ROI_r[2]+ROI_r[3]]
                disp_crop = disp[ROI_d[0]:ROI_d[0]+ROI_d[1],ROI_d[2]:ROI_d[2]+ROI_d[3]]
                disp_scaled_crop = disp_scaled[ROI_d[0]:ROI_d[0]+ROI_d[1],ROI_d[2]:ROI_d[2]+ROI_d[3]]

                # generate 3D points from cropped disparity
                points_crop = self.stereo3D.genDepth(disp_crop)
                # remove inf values
                points_crop[points_crop >= 1E308] = 0

                disp_image_crop_joint = np.concatenate((rect_image_l_crop,disp_scaled_crop,rect_image_r_crop), axis=1)
                cv2.imshow("Stereo 3D Crop", disp_image_crop_joint)

                resPose,squares,detect_img = self.sqDetect.detect_squares(rect_image_l,ROI_l)

                if (resPose):
                    for sq in squares:
                        mask = np.zeros(disp_crop.shape, np.uint8)
                        cv2.drawContours(mask, sq.contour, -1, 255, -1)
                        mean_disp = cv2.mean(disp_crop, mask=mask)[0]
                        x = int(sq.x_pix)
                        y = int(sq.y_pix)
                        depth_xyz = points_crop[y,x]
                        #depth = self.stereo3D.disparity_to_depth(mean_disp)
                        print("Mean: {}".format(mean_disp))
                        print("Point3D @ ({},{}): {}".format(x,y,depth_xyz))
                        print("Point: {},{}".format(sq.x_real,sq.y_real))

                        crop_obj = self.sqDetect.crop_to_contour(disp_scaled_crop,sq.contour)
                        cv2.imshow("Crop object", crop_obj)

                    cv2.imshow("Detected", detect_img)

            k = cv2.waitKey(1)          
            if k == ord('q'): #exit if 'q' key pressed
                self.connected = False
            elif k == ord('s'):
                arrL = np.uint8(rect_image_l)
                h, w = arrL.shape[:2]
                colors = cv2.cvtColor(rect_image_l, cv2.COLOR_BGR2RGB)
                disp_adj = (disp-self.stereo3D.min_disp)/self.stereo3D.num_disp
                mask = disp_adj > disp_adj.min()
                out_points = points[mask]
                out_colors = colors[mask]
                self.stereo3D.write_ply("Data/output.ply",out_points,out_colors)
        print("Closed")

if __name__ == '__main__':
    left_camera_serial = "22864912"
    right_camera_serial = "22864917"
    left_cal_file = "Data/left.yaml"
    right_cal_file = "Data/right.yaml"
    binning = 2

    smartDetect = SmartDetect(left_camera_serial,right_camera_serial,left_cal_file,right_cal_file,binning)
    smartDetect.run()