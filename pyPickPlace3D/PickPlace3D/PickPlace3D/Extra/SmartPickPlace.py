from MoveGroupPythonInteface import MoveGroupPythonInteface
from TISCapture import TISCapture
from BaslerCapture import BaslerCapture
from SquareDetect import SquareDetect, DetectObject
from RobotiqControl import RobotiqControl
from UsefulTranforms import Conversions as tf_c
from StereoCalib import StereoCalib
from Stereo3D import Stereo3D
import numpy as np
import rospy
import cv2
import sys
import math
import urx
import time

class SmartPickPlace():
    def __init__(self,detect_mode="squares",robot_speed=0.05,isUseMoveit=False,binning=1.0,isStereo=True,left_cal_file="",right_cal_file=""):
        self.MOVE_MODE_MOVEIT = "MOVEIT"
        self.MOVE_MODE_URX = "URX"
        if (isUseMoveit):
            self.move_mode = self.MOVE_MODE_MOVEIT
        else:
            self.move_mode = self.MOVE_MODE_URX

        self.detect_mode = detect_mode

        self.robot_vel = robot_speed
        self.robot_acc = robot_speed

        self.isConnected = False
        self.isRobotStopped = True
        self.isPoseValid = False

        self.capture_index = 0

        self.PROCESS_MODE_PICK_PLACE = "PROCESS_MODE_PICK_PLACE"
        self.PROCESS_MODE_CALIBRATE = "PROCESS_MODE_CALIBRATE"
        self.process_mode = self.PROCESS_MODE_CALIBRATE

        self.moveGroupInterface = None
        self.robot_image_zero_position = None
        self.robot_image_zero_q = None
        self.robot_image_zero_orientation = None
        self.binning = binning

        self.sqDetect = SquareDetect(binning=self.binning,isShowThreshold=True)
        self.stCal = StereoCalib("")
        self.isStereo = isStereo

        if (isStereo):
            self.cam2 = BaslerCapture()
            self.stereo3D = Stereo3D(left_cal_file, right_cal_file)
            self.sqDetect.fx = self.stereo3D.fx
            self.sqDetect.baseline= self.stereo3D.T
        else:
            self.cam2 = None
        self.cam = BaslerCapture()

        if (self.move_mode == self.MOVE_MODE_MOVEIT):
            #define robot base to image zero transform
            self.robot_image_zero_position = [0.69888, 0.28909, 0.74546]
            self.robot_image_zero_q = [-0.509778982573, 0.495952045388, 0.491642672896, 0.50243849354]

        elif (self.move_mode == self.MOVE_MODE_URX):
            #define robot base to image zero transform
            #self.robot_image_zero_position = [-0.69000, -0.29291, 0.59005]
            self.robot_image_zero_position = [-0.6015036167497396, -0.3202075274552616, 0.5948835542537181]
            #robot_image_zero_q = [0.99993, 0.00062475, 0.01081, 0.0046892]
            self.robot_image_zero_orientation = [3.14, 0, 3.14]
            self.robot_image_zero_q = tf_c.euler_to_quaternion(self.robot_image_zero_orientation[0],self.robot_image_zero_orientation[1],self.robot_image_zero_orientation[2])

    def connect(self,camera_serial,gripper_port,robot_ip,camera2_serial=None,camera2_delay=0):
        #Connect to camera
        #cam = TISCapture("27810457", 2448, 2048, 35)
        #cam = TISCapture("41810612", 2448, 2048, 35)
        if (self.isStereo):
            if (camera2_serial is None):
                print("Must specify camera2 serial if using stereo camera")
                self.isConnected = False
                return False
            print("Connecting to cameras...")
            ret2 = self.cam2.connect(serial=camera2_serial,binning=self.binning,inter_packet_delay=camera2_delay)
        else:
            print("Connecting to camera...")
            ret2 = True
        ret1 = self.cam.connect(serial=camera_serial,binning=self.binning)

        print("Connecting to robot control...")
        if (self.move_mode == self.MOVE_MODE_MOVEIT):
            self.moveGroupInterface = MoveGroupPythonInteface("manipulator")
            self.moveGroupInterface.group.set_max_velocity_scaling_factor(self.robot_vel)
            self.moveGroupInterface.group.set_max_acceleration_scaling_factor(self.robot_acc)

        elif (self.move_mode == self.MOVE_MODE_URX):
            self.robot = urx.Robot(robot_ip)
            time.sleep(0.2)  #leave some time to robot to process the setup commands

        print("Connecting to gripper...")
        #Connect to gripper 
        self.gripper = RobotiqControl(gripper_port)
        self.gripper.activate()
        self.gripper.open_gripper()

        self.isConnected = True
        print("Systems connected and ready")
        return True

    def run(self):
        while(self.isConnected): #loop if connected (note: only checked at start of program)
            self.process_frame()
        self.close_all()

    def urx_wait_move(self,x,y,z,rr,rp,ry):
        #force tool angle to not wrap wire around itself

        #IF(P18+180>180,-180-(180-(P18+180)),P18+180)
        if (ry+tf_c.d2r(180)>tf_c.d2r(180)):
            ry = -tf_c.d2r(180)-(tf_c.d2r(180)-(ry+tf_c.d2r(180)))
        else:
            ry = ry + tf_c.d2r(180)
        #IF(AND(Q4>-45,Q4<135),Q4+180,Q4)
        if (ry>tf_c.d2r(-45) and ry<tf_c.d2r(135)):
            ry = ry + tf_c.d2r(180)

        #IF(R4>180,-180-(180-R4),R4)
        if (ry>tf_c.d2r(180)):
            ry = tf_c.d2r(-180) - (tf_c.d2r(180)-ry)

        rx, ry, rz = tf_c.rpy2rv(rr,rp,ry)
        while(True):
            self.process_move_frame()
            try:
                if not self.robot.is_program_running():
                    self.robot.movel((x, y, z, rx, ry, rz), self.robot_vel, self.robot_acc,wait=True)
                    time.sleep(0.1)
            except Exception as e:
                print(e)
            if not self.robot.is_program_running():
                print("Robot movement script complete")
                break
        
    def add_robot_offset(self,x,y,z):
        if (self.move_mode == self.MOVE_MODE_MOVEIT):
            offset_x = self.robot_image_zero_position[0] - y
            offset_y = self.robot_image_zero_position[1] - x
        elif(self.move_mode == self.MOVE_MODE_URX):
            offset_x = self.robot_image_zero_position[0] + y
            offset_y = self.robot_image_zero_position[1] + x
        offset_z = self.robot_image_zero_position[2] - z
        return offset_x, offset_y, offset_z

    def create_rect_mat(self,x,y,angle):
        position = [x,y,0]
        orienation = [0,0,angle]
        return tf_c.euler_translation_to_transformation_matrix(orienation,position)

    def place_object(self,x=None,y=None,z=None,qx=None,qy=None,qz=None,qw=None):
        up_offset = 0.05
        resMovePlan = self.move_to_object(False,x,y,z+up_offset,qx,qy,qz,qw)
        resMovePlan = self.move_to_object(False,x,y,z,qx,qy,qz,qw)
        self.gripper.open_gripper()
        resMovePlan = self.move_to_object(False,x,y,z+up_offset,qx,qy,qz,qw)

    def pickup_object(self,x=None,y=None,z=None,qx=None,qy=None,qz=None,qw=None):
        self.gripper.open_gripper()
        up_offset = 0.05
        resMovePlan = self.move_to_object(False,x,y,z+up_offset,qx,qy,qz,qw)
        resMovePlan = self.move_to_object(False,x,y,z,qx,qy,qz,qw)
        self.gripper.close_gripper()
        resMovePlan = self.move_to_object(False,x,y,z+up_offset,qx,qy,qz,qw)

    def move_to_home(self):
        x = self.robot_image_zero_position[0] + 0.4
        y = self.robot_image_zero_position[1]
        z = self.robot_image_zero_position[2]
        qx = self.robot_image_zero_q[0]
        qy = self.robot_image_zero_q[1]
        qz = self.robot_image_zero_q[2]
        qw = self.robot_image_zero_q[3]
        resMovePlan = self.move_to_object(False,x,y,z,qx,qy,qz,qw)
        return resMovePlan

    def move_to_home_init(self):
        x = self.robot_image_zero_position[0] + 0.25
        y = self.robot_image_zero_position[1]
        z = self.robot_image_zero_position[2] + 0.05
        qx = self.robot_image_zero_q[0]
        qy = self.robot_image_zero_q[1]
        qz = self.robot_image_zero_q[2]
        qw = self.robot_image_zero_q[3]
        resMovePlan = self.move_to_object(False,x,y,z,qx,qy,qz,qw)
        return resMovePlan

    def move_to_zero(self):
        x = self.robot_image_zero_position[0]
        y = self.robot_image_zero_position[1]
        z = self.robot_image_zero_position[2]
        qx = self.robot_image_zero_q[0]
        qy = self.robot_image_zero_q[1]
        qz = self.robot_image_zero_q[2]
        qw = self.robot_image_zero_q[3]
        resMovePlan = self.move_to_object(False,x,y,z,qx,qy,qz,qw)
        return resMovePlan

    def execute_plan(self):
        self.moveGroupInterface.group.go(wait=True)

    def move_to_object(self,plan_only=False,x=None,y=None,z=None,qx=None,qy=None,qz=None,qw=None):
        if (self.move_mode == self.MOVE_MODE_URX):
            #get pose and use as target if variables aren't set
            pose_current = self.robot.get_pose()
            if (x == None):
                x = pose_current.pos.x
            if (y == None):
                y = pose_current.pos.y
            if (z == None):
                z = pose_current.pos.z
            #TODO do the same for orienation (pose only returns rotaion matrix so need to convert to quaternion)
            '''
            if (qx == None):
                qx = pose_current.orientation.x
            if (qy == None):
                qy = pose_current.orientation.y
            if (qz == None):
                qz = pose_current.orientation.z
            if (qw == None):
                qw = pose_current.orientation.w
            '''
            if (not plan_only):
                ori = tf_c.quaternion_to_euler(qx, qy, qz,qw)
                self.urx_wait_move(x,y,z,ori[0],ori[1],ori[2])
            return True
        elif (self.move_mode == self.MOVE_MODE_MOVEIT):
            try:
                print "============ Starting Move Group Interface ..."
                pose_current = self.moveGroupInterface.get_current_pose()
                if (x == None):
                    x = pose_current.position.x
                if (y == None):
                    y = pose_current.position.y
                if (z == None):
                    z = pose_current.position.z
                if (qx == None):
                    qx = pose_current.orientation.x
                if (qy == None):
                    qy = pose_current.orientation.y
                if (qz == None):
                    qz = pose_current.orientation.z
                if (qw == None):
                    qw = pose_current.orientation.w

                pose_goal = self.moveGroupInterface.create_pose(x,y,z,qx,qy,qz,qw)
                print("current")
                print(pose_current)

                print("goal")
                print(pose_goal)
                self.isRobotStopped = False
                self.moveGroupInterface.go_to_pose_goal(pose_goal,plan_only)
                self.isRobotStopped = True
                return True
            except rospy.ROSInterruptException as e:
                print(e)
                return False

    def process_frame(self):
        if self.process_mode == self.PROCESS_MODE_PICK_PLACE:
            self.process_detection_frame()
        elif self.process_mode == self.PROCESS_MODE_CALIBRATE:
            self.process_calibration_frame()

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
            elif k == ord('c'): # close gripper
                self.gripper.close_gripper()
            elif k == ord('o'): # open gripper
                self.gripper.open_gripper()
            elif k == ord('h'): # move to home positon
                self.move_to_home()
            elif k == ord('i'): # move to image zero position
                self.move_to_zero()
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
            elif k == ord('1'):
                self.process_mode = self.PROCESS_MODE_CALIBRATE
            elif k == ord('2'):
                self.process_mode = self.PROCESS_MODE_PICK_PLACE

    def coord_to_robot(self,x,y,z,angle):
        z_offset = -1.0
        #calculate object pose relative to robot base
        robot_x, robot_y, robot_z = self.add_robot_offset(x,y,z+z_offset)

        robot_ori = tf_c.quaternion_to_euler(self.robot_image_zero_q[0], self.robot_image_zero_q[1], self.robot_image_zero_q[2],self.robot_image_zero_q[3])
        robot_ori[2] = robot_ori[2] - tf_c.d2r(angle)

        robot_q = tf_c.euler_to_quaternion(robot_ori[0],robot_ori[1],robot_ori[2])
        robot_qx = robot_q[0]
        robot_qy = robot_q[1]
        robot_qz = robot_q[2]
        robot_qw = robot_q[3]

        return robot_x,robot_y,robot_z,robot_qx,robot_qy,robot_qz,robot_qw

    def process_move_frame(self):
        #get image from camera
        resImage,image = self.cam.get_image()
        if (self.isStereo):
            resImage2,image2 = self.cam2.get_image()
        else:
            resImage2 = True
            
        if (resImage and resImage2): #Image received successfully
            #get object position
            if (self.isStereo):
                detect_img = cv2.hconcat([image, image2])
            else:
                detect_img = image
                #resPose,x,y,z,angle,detect_img = self.sqDetect.detect_largest_contour(image)
                
            #print("image angle: {}".format(angle))

            #resize image for displaying
            scale_percent = 40*self.binning # percent of original size
            width = int(detect_img.shape[1] * scale_percent / 100)
            height = int(detect_img.shape[0] * scale_percent / 100)
            resized_image = cv2.resize(detect_img, (width,height), interpolation = cv2.INTER_AREA)

            cv2.imshow('Image Feed', resized_image)

            k = cv2.waitKey(1)        
            if k == ord('q'): #exit if 'q' key pressed
                self.close_all()
        else:
            print("Image capture failed")

    def process_detection_frame(self):
        #get image from camera
        resImage,image = self.cam.get_image()
        if (self.isStereo):
            resImage2,image2 = self.cam2.get_image()
        else:
            resImage2 = True
            
        if (resImage and resImage2): #Image received successfully
            #get object position
            resPose = False
            if (self.isStereo):
                rect_image_l, rect_image_r = self.stereo3D.rectify(image, image2)
                disp = self.stereo3D.gen3D(rect_image_l, rect_image_r)
                
                disp_scaled = self.stereo3D.scale_disparity(disp)
                disp_resized = cv2.resize(disp_scaled, (640, 480), interpolation = cv2.INTER_AREA)

                ROI_l = [626,800,560,1500]
                ROI_r = [626,800,560,1500]
                #187,135
                #374,270
                #ROI_d = [598,800,462,1500]
                ROI_d = ROI_l
                # scale by binning
                ROI_l[:] = [x / self.binning for x in ROI_l]
                ROI_r[:] = [x / self.binning for x in ROI_r]
                #ROI_d[:] = [x / self.binning for x in ROI_d]

                rect_image_l_crop = rect_image_l[ROI_l[0]:ROI_l[0]+ROI_l[1],ROI_l[2]:ROI_l[2]+ROI_l[3]]
                rect_image_r_crop = rect_image_r[ROI_r[0]:ROI_r[0]+ROI_r[1],ROI_r[2]:ROI_r[2]+ROI_r[3]]
                disp_crop = disp[ROI_d[0]:ROI_d[0]+ROI_d[1],ROI_d[2]:ROI_d[2]+ROI_d[3]]
                disp_scaled_crop = disp_scaled[ROI_d[0]:ROI_d[0]+ROI_d[1],ROI_d[2]:ROI_d[2]+ROI_d[3]]

                disp_image_crop_joint = np.concatenate((rect_image_l_crop,disp_scaled_crop,rect_image_r_crop), axis=1)
                cv2.imshow("Stereo 3D Crop", disp_image_crop_joint)

                # run square detection on disparity image
                if (self.detect_mode == "largest"):
                    resPose,x,y,z,angle,detect_img = self.sqDetect.detect_largest_contour(rect_image_l,ROI_l,image2=rect_image_r,ROI2=ROI_r)
                elif (self.detect_mode == "squares"):
                    resPose,squares,detect_img = self.sqDetect.detect_squares(rect_image_l,ROI_l)
                    if (resPose):
                        points_crop = self.stereo3D.genDepth(disp_crop)
                        points_crop[points_crop >= 1E308] = 0
                        sq_idx = 0
                        for sq in squares:
                            mask = np.zeros(disp_crop.shape, np.uint8)
                            cv2.drawContours(mask, sq.contour, -1, 255, -1)
                            mean_disp = cv2.mean(disp_crop, mask=mask)[0]

                            object_crop_image = self.sqDetect.crop_to_contour(disp_scaled_crop,sq.contour)
                            cv2.imshow("object {} (3D)".format(sq_idx),object_crop_image)

                            #mean_xyz = cv2.mean(points_crop, mask=mask)

                            x_pix = int(sq.x_pix)
                            y_pix = int(sq.y_pix)
                            mean_xyz = points_crop[y_pix,x_pix]

                            #xyz_x_offset = -0.025
                            #xyz_y_offset = -0.007
                            #xyz_z_offset = -0.033
                            xyz_x_offset = -0.007
                            xyz_y_offset = -0.007
                            xyz_z_offset = -0.025

                            x_val = mean_xyz[0] + 0.37056866 + xyz_x_offset
                            y_val = mean_xyz[1] + 0.43690515 + xyz_y_offset
                            z_val = mean_xyz[2] + 0.0194396 + xyz_z_offset

                            #depth = self.stereo3D.disparity_to_depth(mean_disp)
                            #print("Mean: {}".format(mean_disp))
                            #print("Point3D @ ({},{}): {}".format(x_pix,y_pix,mean_xyz))
                            print("Point3D @ ({},{}): {}".format(0,0,points_crop[0,0]))
                            #print("Point: {},{}".format(sq.x_real,sq.y_real))

                            if (z_val > 1.3):
                                z_val = 1.3
                            elif (z_val < 0.9):
                                z_val = 0.9
                            squares[sq_idx].x_real = x_val
                            squares[sq_idx].y_real = y_val
                            squares[sq_idx].z_real = z_val

                            print("XYZ{}: {},{},{}".format(sq_idx,squares[sq_idx].x_real,squares[sq_idx].y_real,squares[sq_idx].z_real))

                            #crop_obj = self.sqDetect.crop_to_contour(disp_scaled_crop,sq.contour)
                            #cv2.imshow("Crop object", crop_obj)
                            sq_idx+=1
                        
            else:
                if (self.detect_mode == "largest"):
                    resPose,x,y,z,angle,detect_img = self.sqDetect.detect_largest_contour(image)
                elif (self.detect_mode == "squares"):
                    resPose,squares,detect_img = self.sqDetect.detect_squares(image)
                
            grip_offset = 0.015

            if resPose:
                #resize image for displaying
                scale_percent = 60*self.binning # percent of original size
                width = int(detect_img.shape[1] * scale_percent / 100)
                height = int(detect_img.shape[0] * scale_percent / 100)
                resized_image = cv2.resize(detect_img, (width,height), interpolation = cv2.INTER_AREA)

                cv2.imshow('Image Feed', resized_image)

            k = cv2.waitKey(1)        
            if k == ord('q'): #exit if 'q' key pressed
                self.close_all()
            elif k == ord('c'): # close gripper
                self.gripper.close_gripper()
            elif k == ord('o'): # open gripper
                self.gripper.open_gripper()
            elif k == ord('h'): # move to home positon
                self.move_to_home_init()
                self.move_to_home()
            elif k == ord('d'): # get the data on the objects position
                if (resPose):
                    if (self.detect_mode == "squares"):
                        x = squares[0].x_real
                        y = squares[0].y_real
                        z = squares[0].z_real
                        angle = squares[0].angle
                    area = cv2.contourArea(squares[0].contour)

                    move_x, move_y, move_z, move_qx, move_qy, move_qz, move_qw  = self.coord_to_robot(x,y,z,angle)
                    print("image_frame: {},{},{} {}".format(x, y, z, angle))
                    print("robot_frame: ({},{},{}) ({},{},{},{})".format(move_x, move_y, move_z, move_qx, move_qy, move_qz, move_qw))
            elif k == ord('t'): # move above object
                if (resPose):
                    if (self.detect_mode == "squares"):
                        x = squares[0].x_real
                        y = squares[0].y_real
                        z = squares[0].z_real
                        angle = squares[0].angle

                        print("Moving to: {},{},{}".format(x,y,z))

                    move_x, move_y, move_z, move_qx, move_qy, move_qz, move_qw  = self.coord_to_robot(x,y,z,angle)
                    self.gripper.close_gripper()
                    self.move_to_home_init()
                    up_offset = 0.05
                    resMovePlan = self.move_to_object(False,move_x,move_y,move_z + up_offset,move_qx,move_qy,move_qz,move_qw)
                    resMovePlan = self.move_to_object(False,move_x,move_y,move_z,move_qx,move_qy,move_qz,move_qw)
                    '''
                    resMovePlan = self.move_to_object(False,move_x,move_y,move_z + up_offset,move_qx,move_qy,move_qz,move_qw)
                    self.move_to_home_init()
                    self.move_to_home()
                    '''
            elif k == ord('g'): # pickup object
                if (resPose):
                    if (self.detect_mode == "squares"):
                        x = squares[0].x_real
                        y = squares[0].y_real
                        z = squares[0].z_real + grip_offset
                        angle = squares[0].angle
                    
                    move_x, move_y, move_z, move_qx, move_qy, move_qz, move_qw  = self.coord_to_robot(x,y,z,angle)
                    self.move_to_home_init()
                    self.pickup_object(move_x,move_y,move_z,move_qx,move_qy,move_qz,move_qw)
                    self.place_object(move_x,move_y,move_z,move_qx,move_qy,move_qz,move_qw)
                    self.move_to_home_init()
                    self.move_to_home()
            elif k == ord('p'): # pickup objects and place in pattern
                if (resPose):
                    if (self.detect_mode == "squares"):
                        large_block_area_max = 40000 / self.binning
                        small_block_area_max = 15000 / self.binning
                        small_block_area_min = 5000 / self.binning

                        print("Large block range: {} - {}".format(small_block_area_max,large_block_area_max))
                        print("Small block range: {} - {}".format(small_block_area_min,small_block_area_max))
                        if (len(squares) == 4):
                            # load squares list into local list
                            squares_n = []
                            for sq in squares:
                                squares_n.append(sq)

                            large_blocks = []
                            small_blocks = []
                            print("checking valid block areas")
                            for sq in squares_n:
                                # get contour area
                                contour_area = cv2.contourArea(sq.contour)
                                print(contour_area)
                                if (contour_area < large_block_area_max):
                                    if (contour_area < small_block_area_max and contour_area > small_block_area_min):
                                        #is small block
                                        small_blocks.append(sq)
                                    else:
                                        #is large block
                                        large_blocks.append(sq)
                                else:
                                    #is too large
                                    print("Contour is outside of limits for valid blocks")

                            if (len(large_blocks) == 2 and len(small_blocks) == 2):
                                # get position of first large object
                                x = large_blocks[0].x_real
                                y = large_blocks[0].y_real
                                z = large_blocks[0].z_real + grip_offset
                                angle = large_blocks[0].angle
                                # calculate position in robot frame
                                move_x, move_y, move_z, move_qx, move_qy, move_qz, move_qw  = self.coord_to_robot(x,y,z,angle)
                                # pickup object
                                self.move_to_home_init()
                                self.pickup_object(move_x,move_y,move_z,move_qx,move_qy,move_qz,move_qw)
                                
                                # move to first large position
                                new_pos = [0.602717179108,0.148549647903,0.995,-90]
                                new_x, new_y, new_z, new_angle = [new_pos[i] for i in (0, 1, 2, 3)]
                                #new_z = new_z - above_part_offset
                                print("new_image_frame: {},{},{} {}".format(x, y, z, angle))
                                move_x, move_y, move_z, move_qx, move_qy, move_qz, move_qw  = self.coord_to_robot(new_x, new_y, new_z, new_angle)
                                # place object
                                self.place_object(move_x,move_y,move_z,move_qx,move_qy,move_qz,move_qw)

                                # get position of second large object
                                x = large_blocks[1].x_real
                                y = large_blocks[1].y_real
                                z = large_blocks[1].z_real + grip_offset
                                angle = large_blocks[1].angle
                                # calculate position in robot frame
                                move_x, move_y, move_z, move_qx, move_qy, move_qz, move_qw  = self.coord_to_robot(x,y,z,angle)
                                # pickup object
                                self.pickup_object(move_x,move_y,move_z,move_qx,move_qy,move_qz,move_qw)

                                # move to second large position
                                new_pos = [0.488542124939,0.148549647903,1.00,-89.8]
                                new_x, new_y, new_z, new_angle = [new_pos[i] for i in (0, 1, 2, 3)]
                                #new_z = new_z - above_part_offset
                                print("new_image_frame: {},{},{} {}".format(x, y, z, angle))
                                move_x, move_y, move_z, move_qx, move_qy, move_qz, move_qw  = self.coord_to_robot(new_x, new_y, new_z, new_angle)
                                # place object
                                self.place_object(move_x,move_y,move_z,move_qx,move_qy,move_qz,move_qw)

                                # get position of first small object
                                x = small_blocks[0].x_real
                                y = small_blocks[0].y_real
                                z = small_blocks[0].z_real + grip_offset
                                angle = small_blocks[0].angle
                                # calculate position in robot frame
                                move_x, move_y, move_z, move_qx, move_qy, move_qz, move_qw  = self.coord_to_robot(x,y,z,angle)
                                # pickup object
                                self.pickup_object(move_x,move_y,move_z,move_qx,move_qy,move_qz,move_qw)

                                # move to first small position
                                new_pos = [0.545709375,0.239640625,1.00,-0.0]
                                new_x, new_y, new_z, new_angle = [new_pos[i] for i in (0, 1, 2, 3)]
                                #new_z = new_z + above_part_offset
                                print("new_image_frame: {},{},{} {}".format(x, y, z, angle))
                                move_x, move_y, move_z, move_qx, move_qy, move_qz, move_qw  = self.coord_to_robot(new_x, new_y, new_z, new_angle)
                                # place object
                                self.place_object(move_x,move_y,move_z,move_qx,move_qy,move_qz,move_qw)

                                # get position of second small object
                                x = small_blocks[1].x_real
                                y = small_blocks[1].y_real
                                z = small_blocks[1].z_real + grip_offset
                                angle = small_blocks[1].angle
                                # calculate position in robot frame
                                move_x, move_y, move_z, move_qx, move_qy, move_qz, move_qw  = self.coord_to_robot(x,y,z,angle)
                                # pickup object
                                self.pickup_object(move_x,move_y,move_z,move_qx,move_qy,move_qz,move_qw)
                                
                                # move to second small position
                                new_pos = [0.545709375,0.058553125,1.00,-0.0]
                                new_x, new_y, new_z, new_angle = [new_pos[i] for i in (0, 1, 2, 3)]
                                #new_z = new_z + above_part_offset
                                print("new_image_frame: {},{},{} {}".format(x, y, z, angle))
                                move_x, move_y, move_z, move_qx, move_qy, move_qz, move_qw  = self.coord_to_robot(new_x, new_y, new_z, new_angle)
                                # place object
                                self.place_object(move_x,move_y,move_z,move_qx,move_qy,move_qz,move_qw)

                                self.move_to_home_init()
                                self.move_to_home()
                            else:
                                print("invalid number of large and small objects")
                        else:
                            print("invalid number of objects")
                    else:
                        print("This process is only possible if detect_mode='squares'")
            elif k == ord('1'):
                self.process_mode = self.PROCESS_MODE_CALIBRATE
            elif k == ord('2'):
                self.process_mode = self.PROCESS_MODE_PICK_PLACE
        else:
            print("Image capture failed")

    def close_all(self):
        print("Quitting...")
        print("Closing gripper connection...")
        self.gripper.close()
        print("Closing camera connection...")
        self.cam.close()
        if (self.move_mode == self.MOVE_MODE_URX):
            print("Closing robot connection...")
            self.robot.close()
        print("All systems closed")
        cv2.destroyAllWindows()
        exit()

if __name__ == '__main__':
    isStereo = True
    binning = 2
    robot_ip = "172.16.1.79"
    robot_speed = 0.1
    gripper_port = "/dev/ttyUSB0"
    left_camera_serial = "22864912"
    right_camera_serial = "22864917"
    left_cal_file = "Data/left.yaml"
    right_cal_file = "Data/right.yaml"
    detect_mode = "squares" # "squares"/"largest"

    if (isStereo):
        smartpp = SmartPickPlace(detect_mode=detect_mode,robot_speed=robot_speed,binning=binning,
                                    isStereo=True,left_cal_file=left_cal_file,right_cal_file=right_cal_file)
        cam2_delay = 5000
        if (binning > 1):
            cam2_delay = 0
        smartpp.connect(left_camera_serial,gripper_port,robot_ip,right_camera_serial,camera2_delay=cam2_delay)
    else:
        smartpp = SmartPickPlace(detect_mode=detect_mode,robot_speed=robot_speed,binning=binning,isStereo=False)
        smartpp.connect(left_camera_serial,gripper_port,robot_ip)
    
    smartpp.process_mode = smartpp.PROCESS_MODE_PICK_PLACE
    smartpp.run()