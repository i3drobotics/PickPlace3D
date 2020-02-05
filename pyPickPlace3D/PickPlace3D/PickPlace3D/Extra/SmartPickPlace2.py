from Stereo3D import Stereo3D
from BaslerCapture import BaslerCapture
from SquareDetect import SquareDetect, DetectObject
from RobotiqControl import RobotiqControl
from UsefulTranforms import Conversions as tf_c
import urx
import numpy as np
import cv2
import time
import math

class SmartPickPlace2():
    def __init__(self,robot_ip,robot_speed,gripper_ip,left_camera_serial,right_camera_serial,left_cal_file,right_cal_file,binning):
        self.connected = False
        self.binning = binning
        self.cam_l = BaslerCapture()
        self.cam_r = BaslerCapture()
        self.robot = urx.Robot(robot_ip)
        self.robot_vel = robot_speed
        self.robot_acc = robot_speed
        self.gripper = RobotiqControl(gripper_port)
        self.sqDetect = SquareDetect(binning=self.binning,isShowThreshold=False)
        print("Loading calibration...")
        self.stereo3D = Stereo3D(left_cal_file, right_cal_file)
        self.camera_calibrated = False
        self.robot_calibrated = False
        self.image_zero = [-1,-1]
        self.image_zero_anti_binning = [-1,-1]
        self.robot_zero = [-0.6852423340444724, -0.26550360706314924, 0.5740333751316029]
        self.robot_zero_orientation = [3.14, 0, 3.14]
        self.robot_zero_q = tf_c.euler_to_quaternion(self.robot_zero_orientation[0],self.robot_zero_orientation[1],self.robot_zero_orientation[2])
        
    def connect(self):
        self.gripper.activate()
        self.gripper.open_gripper()
        if (self.stereo3D.isCalLoaded()):
            #connect to cameras
            print("Connecting to cameras...")
            ret_l = self.cam_l.connect(left_camera_serial,binning=binning)
            ret_r = self.cam_r.connect(right_camera_serial,binning=binning)
            if (ret_l and ret_r):
                self.connected = True
                return True
        return False

    def place_object(self,x=None,y=None,z=None,qx=None,qy=None,qz=None,qw=None):
        up_offset = 0.05
        resMovePlan = self.move_to_position(x,y,z+up_offset,qx,qy,qz,qw)
        resMovePlan = self.move_to_position(x,y,z,qx,qy,qz,qw)
        self.gripper.open_gripper()
        resMovePlan = self.move_to_position(x,y,z+up_offset,qx,qy,qz,qw)

    def pickup_object(self,x=None,y=None,z=None,qx=None,qy=None,qz=None,qw=None):
        self.gripper.open_gripper()
        up_offset = 0.05
        resMovePlan = self.move_to_position(x,y,z+up_offset,qx,qy,qz,qw)
        resMovePlan = self.move_to_position(x,y,z,qx,qy,qz,qw)
        self.gripper.close_gripper()
        resMovePlan = self.move_to_position(x,y,z+up_offset,qx,qy,qz,qw)

    def move_to_home(self):
        x = self.robot_zero[0] + 0.4
        y = self.robot_zero[1] 
        z = self.robot_zero[2]
        qx = self.robot_zero_q[0]
        qy = self.robot_zero_q[1]
        qz = self.robot_zero_q[2]
        qw = self.robot_zero_q[3]
        resMovePlan = self.move_to_position(x,y,z,qx,qy,qz,qw)
        return resMovePlan

    def move_to_home_init(self):
        x = self.robot_zero[0] + 0.25
        y = self.robot_zero[1]
        z = self.robot_zero[2]
        qx = self.robot_zero_q[0]
        qy = self.robot_zero_q[1]
        qz = self.robot_zero_q[2]
        qw = self.robot_zero_q[3]
        resMovePlan = self.move_to_position(x,y,z,qx,qy,qz,qw)
        return resMovePlan

    def move_to_position(self,x,y,z,qx,qy,qz,qw):
        ori = tf_c.quaternion_to_euler(qx, qy, qz,qw)
        self.urx_wait_move(x,y,z,ori[0],ori[1],ori[2])
    
    def add_robot_offset(self,x,y,z):
        offset_x = self.robot_zero[0] + y
        offset_y = self.robot_zero[1] + x
        offset_z = self.robot_zero[2] - z
        return offset_x, offset_y, offset_z

    def coord_to_robot(self,x,y,z,angle):
        z_offset = -1.0
        #calculate object pose relative to robot base
        robot_x, robot_y, robot_z = self.add_robot_offset(x,y,z+z_offset)

        robot_ori = tf_c.quaternion_to_euler(self.robot_zero_q[0], self.robot_zero_q[1], self.robot_zero_q[2],self.robot_zero_q[3])
        robot_ori[2] = robot_ori[2] - tf_c.d2r(angle)

        robot_q = tf_c.euler_to_quaternion(robot_ori[0],robot_ori[1],robot_ori[2])
        robot_qx = robot_q[0]
        robot_qy = robot_q[1]
        robot_qz = robot_q[2]
        robot_qw = robot_q[3]

        return robot_x,robot_y,robot_z,robot_qx,robot_qy,robot_qz,robot_qw

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
            self.process_frame()
            try:
                if not self.robot.is_program_running():
                    self.robot.movel((x, y, z, rx, ry, rz), self.robot_vel, self.robot_acc,wait=True)
                    time.sleep(0.1)
            except Exception as e:
                print(e)
            if not self.robot.is_program_running():
                print("Robot movement script complete")
                break

    def click_position(self,event,x,y,flags,para):
        if event == cv2.EVENT_LBUTTONDOWN:
            bin_x = x * self.binning
            bin_y = y * self.binning
            print([(bin_x, bin_y)])

    def pick_up_pattern(self,squares):
        grip_offset = -0.015
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
                z = large_blocks[0].z_real
                angle = large_blocks[0].angle
                # calculate position in robot frame
                move_x, move_y, move_z, move_qx, move_qy, move_qz, move_qw  = self.coord_to_robot(x,y,z,angle)
                # pickup object
                self.move_to_home_init()
                self.pickup_object(move_x,move_y,move_z + grip_offset,move_qx,move_qy,move_qz,move_qw)
                
                # move to first large position
                new_pos = [0.606717179108,0.148549647903,0.997,-90]
                new_x, new_y, new_z, new_angle = [new_pos[i] for i in (0, 1, 2, 3)]
                #new_z = new_z - above_part_offset
                print("new_image_frame: {},{},{} {}".format(x, y, z, angle))
                move_x, move_y, move_z, move_qx, move_qy, move_qz, move_qw  = self.coord_to_robot(new_x, new_y, new_z, new_angle)
                # place object
                self.place_object(move_x,move_y,move_z,move_qx,move_qy,move_qz,move_qw)

                # get position of second large object
                x = large_blocks[1].x_real
                y = large_blocks[1].y_real
                z = large_blocks[1].z_real
                angle = large_blocks[1].angle
                # calculate position in robot frame
                move_x, move_y, move_z, move_qx, move_qy, move_qz, move_qw  = self.coord_to_robot(x,y,z,angle)
                # pickup object
                self.pickup_object(move_x,move_y,move_z + grip_offset,move_qx,move_qy,move_qz,move_qw)

                # move to second large position
                new_pos = [0.488542124939,0.148549647903,1.002,-90]
                new_x, new_y, new_z, new_angle = [new_pos[i] for i in (0, 1, 2, 3)]
                #new_z = new_z - above_part_offset
                print("new_image_frame: {},{},{} {}".format(x, y, z, angle))
                move_x, move_y, move_z, move_qx, move_qy, move_qz, move_qw  = self.coord_to_robot(new_x, new_y, new_z, new_angle)
                # place object
                self.place_object(move_x,move_y,move_z,move_qx,move_qy,move_qz,move_qw)

                # get position of first small object
                x = small_blocks[0].x_real
                y = small_blocks[0].y_real
                z = small_blocks[0].z_real
                angle = small_blocks[0].angle
                # calculate position in robot frame
                move_x, move_y, move_z, move_qx, move_qy, move_qz, move_qw  = self.coord_to_robot(x,y,z,angle)
                # pickup object
                self.pickup_object(move_x,move_y,move_z + grip_offset,move_qx,move_qy,move_qz,move_qw)

                # move to first small position
                new_pos = [0.545709375,0.239640625,1.003,-0.0]
                new_x, new_y, new_z, new_angle = [new_pos[i] for i in (0, 1, 2, 3)]
                #new_z = new_z + above_part_offset
                print("new_image_frame: {},{},{} {}".format(x, y, z, angle))
                move_x, move_y, move_z, move_qx, move_qy, move_qz, move_qw  = self.coord_to_robot(new_x, new_y, new_z, new_angle)
                # place object
                self.place_object(move_x,move_y,move_z,move_qx,move_qy,move_qz,move_qw)

                # get position of second small object
                x = small_blocks[1].x_real
                y = small_blocks[1].y_real
                z = small_blocks[1].z_real
                angle = small_blocks[1].angle
                # calculate position in robot frame
                move_x, move_y, move_z, move_qx, move_qy, move_qz, move_qw  = self.coord_to_robot(x,y,z,angle)
                # pickup object
                self.pickup_object(move_x,move_y,move_z + grip_offset,move_qx,move_qy,move_qz,move_qw)
                
                # move to second small position
                new_pos = [0.545709375,0.058553125,0.997,-0.0]
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

    def process_frame(self):
        res3D = False
        res_img_l,image_l = self.cam_l.get_image()
        res_img_r,image_r = self.cam_r.get_image()
        if (res_img_l and res_img_r):
            rect_image_l, rect_image_r = self.stereo3D.rectify(image_l, image_r)

            rect_image_l_resize = cv2.resize(rect_image_l, (640, 480), interpolation = cv2.INTER_AREA)
            rect_image_r_resize = cv2.resize(rect_image_r, (640, 480), interpolation = cv2.INTER_AREA)

            disp = self.stereo3D.gen3D(rect_image_l, rect_image_r)

            disp_scaled = self.stereo3D.scale_disparity(disp)
            disp_resized = cv2.resize(disp_scaled, (640, 480), interpolation = cv2.INTER_AREA)
            
            # generate 3D points from disparity
            points = self.stereo3D.genDepth(disp)
            # remove inf values
            points[points >= 1E308] = 0

            res3D = True

            ROI = None
            if (self.image_zero_anti_binning[0] == -1):
                ROI = [616,800,560,1500]
            else:
                ROI = [self.image_zero_anti_binning[1],800,self.image_zero_anti_binning[0],1500]
            ROI[:] = [x / self.binning for x in ROI]

            rect_image_l_crop = rect_image_l[ROI[0]:ROI[0]+ROI[1],ROI[2]:ROI[2]+ROI[3]]
            rect_image_r_crop = rect_image_r[ROI[0]:ROI[0]+ROI[1],ROI[2]:ROI[2]+ROI[3]]
            disp_crop = disp[ROI[0]:ROI[0]+ROI[1],ROI[2]:ROI[2]+ROI[3]]
            disp_scaled_crop = disp_scaled[ROI[0]:ROI[0]+ROI[1],ROI[2]:ROI[2]+ROI[3]]

            resPose,squares,detect_img = self.sqDetect.detect_squares(rect_image_l,ROI)

            #detect_img_resized = cv2.resize(detect_img, (640, 480), interpolation = cv2.INTER_AREA)
            disp_detect_image_joint = np.concatenate((rect_image_l_resize,disp_resized), axis=1)
            cv2.imshow("Detection", disp_detect_image_joint)

        k = cv2.waitKey(1)          
        if k == ord('q'): # exit if 'q' key pressed
            self.connected = False
            self.close_all()

    def process_run(self):
        res3D = False
        res_img_l,image_l = self.cam_l.get_image()
        res_img_r,image_r = self.cam_r.get_image()
        if (res_img_l and res_img_r):
            rect_image_l, rect_image_r = self.stereo3D.rectify(image_l, image_r)

            rect_image_l_resize = cv2.resize(rect_image_l, (640, 480), interpolation = cv2.INTER_AREA)
            rect_image_r_resize = cv2.resize(rect_image_r, (640, 480), interpolation = cv2.INTER_AREA)

            disp = self.stereo3D.gen3D(rect_image_l, rect_image_r)

            disp_scaled = self.stereo3D.scale_disparity(disp)
            disp_resized = cv2.resize(disp_scaled, (640, 480), interpolation = cv2.INTER_AREA)
            
            # generate 3D points from disparity
            points = self.stereo3D.genDepth(disp)
            # remove inf values
            points[points >= 1E308] = 0

            res3D = True

            ROI = None
            if (self.image_zero_anti_binning[0] == -1):
                ROI = [616,800,560,1500]
            else:
                ROI = [self.image_zero_anti_binning[1],800,self.image_zero_anti_binning[0],1500]
            ROI[:] = [x / self.binning for x in ROI]

            rect_image_l_crop = rect_image_l[ROI[0]:ROI[0]+ROI[1],ROI[2]:ROI[2]+ROI[3]]
            rect_image_r_crop = rect_image_r[ROI[0]:ROI[0]+ROI[1],ROI[2]:ROI[2]+ROI[3]]
            disp_crop = disp[ROI[0]:ROI[0]+ROI[1],ROI[2]:ROI[2]+ROI[3]]
            disp_scaled_crop = disp_scaled[ROI[0]:ROI[0]+ROI[1],ROI[2]:ROI[2]+ROI[3]]

            resPose,squares,detect_img = self.sqDetect.detect_squares(rect_image_l,ROI)

            detect_img_resized = cv2.resize(detect_img, (640, 480), interpolation = cv2.INTER_AREA)
            disp_detect_image_joint = np.concatenate((detect_img_resized,disp_resized), axis=1)
            cv2.imshow("Detection", disp_detect_image_joint)

            if (resPose):
                points_crop = self.stereo3D.genDepth(disp_crop)
                points_crop[points_crop >= 1E308] = 0
                sq_idx = 0
                for sq in squares:
                    mask = np.zeros(disp_crop.shape, np.uint8)
                    cv2.drawContours(mask, sq.contour, -1, 255, -1)
                    mean_disp = cv2.mean(disp_crop, mask=mask)[0]

                    object_crop_image = self.sqDetect.crop_to_contour(disp_scaled_crop,sq.contour)
                    #cv2.imshow("object {} (3D)".format(sq_idx),object_crop_image)

                    #mean_xyz = cv2.mean(points_crop, mask=mask)

                    x_pix = int(sq.x_pix)
                    y_pix = int(sq.y_pix)
                    mean_xyz = points_crop[y_pix,x_pix]

                    #xyz_x_offset = -0.00459647283811
                    #xyz_y_offset = -0.00444763557421
                    #xyz_z_offset = -0.0083806911437
                    xyz_x_offset = -0.00642008331126
                    xyz_y_offset = -0.0048256068904
                    xyz_z_offset = -0.0087283516676

                    x_val = mean_xyz[0] - points_crop[0,0][0] + xyz_x_offset
                    y_val = mean_xyz[1] - points_crop[0,0][1] + xyz_y_offset

                    # tan(slop_angle) x Y
                    slop_angle = 3
                    slop_offset = math.atan(math.radians(slop_angle)) * y_val

                    z_val = mean_xyz[2] + slop_offset + xyz_z_offset               

                    #depth = self.stereo3D.disparity_to_depth(mean_disp)
                    #print("Mean: {}".format(mean_disp))
                    #print("Point3D @ ({},{}): {}".format(x_pix,y_pix,mean_xyz))
                    #print("Point3D @ ({},{}): {}".format(0,0,points_crop[0,0]))
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

        k = cv2.waitKey(1)          
        if k == ord('q'): # exit if 'q' key pressed
            self.connected = False
            self.close_all()
        elif k == ord('c'): # close gripper
            self.gripper.close_gripper()
        elif k == ord('o'): # open gripper
            self.gripper.open_gripper()
        elif k == ord('h'): # move robot to home
            print("Moving to home")
            self.move_to_home()
        elif k == ord('p'):
            self.pick_up_pattern(squares)
        elif k == ord('g'):
            if (resPose):
                x = squares[0].x_real
                y = squares[0].y_real
                z = squares[0].z_real
                angle = squares[0].angle
                
                move_x, move_y, move_z, move_qx, move_qy, move_qz, move_qw  = self.coord_to_robot(x,y,z,angle)
                self.move_to_home_init()
                grip_offset = -0.015
                self.pickup_object(move_x,move_y,move_z+grip_offset,move_qx,move_qy,move_qz,move_qw)
                self.place_object(move_x,move_y,move_z+grip_offset,move_qx,move_qy,move_qz,move_qw)
                self.move_to_home_init()
                self.move_to_home()
        elif k == ord('t'):
            if (resPose):
                x = squares[0].x_real
                y = squares[0].y_real
                z = squares[0].z_real
                angle = squares[0].angle

                print("Moving to: {},{},{}".format(x,y,z))

                move_x, move_y, move_z, move_qx, move_qy, move_qz, move_qw  = self.coord_to_robot(x,y,z,angle)
                self.gripper.close_gripper()
                self.move_to_home()
                up_offset = 0.05
                resMovePlan = self.move_to_position(move_x,move_y,move_z + up_offset,move_qx,move_qy,move_qz,move_qw)
                resMovePlan = self.move_to_position(move_x,move_y,move_z,move_qx,move_qy,move_qz,move_qw)
        elif k == ord('d'): # detect qr code if 'd' key pressed
            cxs,cys = self.stereo3D.detect_aruco(rect_image_l,0.05)
            if (len(cxs) > 0):
                self.image_zero = [cxs[0],cys[0]]
                self.image_zero_anti_binning = [cxs[0]*self.binning,cys[0]*self.binning]
                print("Storing marker position - X: {}, Y: {} as image zero".format(self.image_zero[0],self.image_zero[1]))
                print("Storing marker position - X: {}, Y: {} as image zero (adjusted for binning)".format(self.image_zero_anti_binning[0],self.image_zero_anti_binning[1]))
        elif k == ord('r'): # store robot position if 'r' key pressed as robot zero
            robot_xyzrpy = self.robot.getl()
            print "Current tool pose is: ",  robot_xyzrpy
            self.robot_zero = [robot_xyzrpy[0],robot_xyzrpy[1],robot_xyzrpy[2]]
            print "Storing current tool pose - X: {}, Y: {}, Z {} as robot zero".format(self.robot_zero[0],self.robot_zero[1],self.robot_zero[2])
        elif k == ord('1'): # store robot position if '1' key pressed as position 1
            robot_xyzrpy = self.robot.getl()
            print "Current tool pose is: ",  robot_xyzrpy
            self.robot_p1 = [robot_xyzrpy[0],robot_xyzrpy[1],robot_xyzrpy[2]]
            print "Storing current tool pose - X: {}, Y: {}, Z {} as robot p1".format(self.robot_p1[0],self.robot_p1[1],self.robot_p1[2])
        elif k == ord('2'): # store robot position if '2' key pressed and calculate distance from position 1
            robot_xyzrpy = self.robot.getl()
            print "Current tool pose is: ",  robot_xyzrpy
            self.robot_p2 = [robot_xyzrpy[0],robot_xyzrpy[1],robot_xyzrpy[2]]
            print "Storing current tool pose - X: {}, Y: {}, Z {} as robot p2".format(self.robot_p2[0],self.robot_p2[1],self.robot_p2[2])
            diff = []
            for p1, p2, in zip(self.robot_p1,self.robot_p2):
                diff.append(p2-p1)
            print "Positional difference from p1 - X: {}, Y: {}, Z {}".format(diff[0],diff[1],diff[2])

    def run(self):
        while (self.connected):
            self.process_run()
        print("Closed")
        self.close_all()

    def close_all(self):
        print("Quitting...")
        print("Closing gripper connection...")
        self.gripper.close()
        print("Closing camera connections...")
        self.cam_l.close()
        self.cam_r.close()
        print("Closing robot connection...")
        self.robot.close()
        print("All systems closed")
        cv2.destroyAllWindows()
        exit()

if __name__ == '__main__':
    robot_ip = "172.16.1.79"
    robot_speed = 0.15
    gripper_port = "/dev/ttyUSB0"
    left_camera_serial = "22864912"
    right_camera_serial = "22864917"
    left_cal_file = "Data/left.yaml"
    right_cal_file = "Data/right.yaml"
    binning = 2

    smartPickPlace = SmartPickPlace2(robot_ip,robot_speed,gripper_port,left_camera_serial,right_camera_serial,left_cal_file,right_cal_file,binning)
    smartPickPlace.connect()
    smartPickPlace.run()