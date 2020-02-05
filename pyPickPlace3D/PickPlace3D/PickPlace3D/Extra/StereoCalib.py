import numpy as np
import cv2

class StereoCalib():
    def __init__(self,folderpath,square_width=0.038,grid_rows=6,grid_cols=8):
        self.square_width = square_width
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.folderpath = folderpath
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.reset()

    def find_checkerboard(self,img):
        # convert to grey image
        #img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_grey = img
        # find checkboard in image
        ret, corners = cv2.findChessboardCorners(img_grey, (self.grid_cols,self.grid_rows),None)
        if (ret):
            cv2.cornerSubPix(img_grey,corners,(11,11),(-1,-1),self.criteria)
        return ret, img_grey, corners

    def reset(self):
        print("Clearing calibration...")
        self.isReady = False
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((self.grid_rows*self.grid_cols,3), np.float32)
        self.objp[:,:2] = np.mgrid[0:self.grid_cols,0:self.grid_rows].T.reshape(-1,2)
        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d point in real world space
        self.img1points = [] # 2d points in image plane.
        self.img2points = [] # 2d points in image plane.
        self.image_left = None
        self.image_right = None
        self.capture_valid = False
        self.capture_index = 0

        self.left_rectification = None
        self.right_rectificaiton = None
        self.left_projection = None
        self.right_projection = None
        self.left_distortion = None
        self.right_distortion = None
        self.left_camera_matrix = None
        self.right_camera_matrix = None

        print("Calibration reset")

    def add_frame(self):
        if (self.capture_valid):
            self.capture_index += 1
            # Add object and image points to array for processing later
            self.objpoints.append(self.objp)
            self.img1points.append(self.corners1)
            self.img2points.append(self.corners2)
            print("Stereo image %d grabbed. Press 't' to run calibration.",self.capture_index)
            return True
        else:
            print("last image pair received was not processed sucessfully")
            return False

    def process_frame(self,img1,img2):
        # find checkboard in each image
        ret1, img_grey1, corners1 = self.find_checkerboard(img1)
        if (ret1):
            ret2, img_grey2, corners2 = self.find_checkerboard(img2)
        # check checkboard was found in both images
        if (ret1 and ret2):
            self.image_left = img_grey1
            self.image_right = img_grey2
            self.corners1 = corners1
            self.corners2 = corners2
            self.image_size = self.image_left.shape[::-1]
            self.capture_valid = True

            # Draw and display the corners
            cv2.drawChessboardCorners(img1, (self.grid_cols,self.grid_rows), corners1, ret1)
            cv2.drawChessboardCorners(img2, (self.grid_cols,self.grid_rows), corners2, ret2)

            img_join = cv2.hconcat([img1, img2])

            return True,img_join
        else:
            self.capture_valid = False
            img_join = cv2.hconcat([img1, img2])

            return False,img_join

    def rectify(self,img1,img2):
        leftMapX, leftMapY = cv2.initUndistortRectifyMap(
            self.left_camera_matrix, self.left_distortion, self.left_rectification,
            self.left_projection, self.image_size, cv2.CV_32FC1)

        rightMapX, rightMapY = cv2.initUndistortRectifyMap(
            self.right_camera_matrix, self.right_distortion, self.right_rectificaiton,
            self.right_projection, self.image_size, cv2.CV_32FC1)

        img_rect1 = cv2.remap(img1, leftMapX, leftMapY, cv2.INTER_LINEAR)
        img_rect2 = cv2.remap(img2, rightMapX, rightMapY, cv2.INTER_LINEAR)
        
        return img_rect1, img_rect2

    def save_calibration(self,folderpath):
        print("Saving calibration to folder %s, in files left.yaml and right.yaml...",folderpath)
        fs_write = cv2.FileStorage(folderpath+"/left.yaml", cv2.FILE_STORAGE_WRITE)
        fs_write.write("image_width", int(self.image_size[1]))
        fs_write.write("image_height", int(self.image_size[0]))
        fs_write.write("camera_name", "leftCamera")
        fs_write.write("camera_matrix", self.left_camera_matrix)
        fs_write.write("distortion_model", "plumb_bob")
        fs_write.write("distortion_coefficients", self.left_distortion)
        fs_write.write("rectification_matrix", self.left_rectification)
        fs_write.write("projection_matrix", self.left_projection)
        fs_write.release()

        fs_write = cv2.FileStorage(folderpath+"/right.yaml", cv2.FILE_STORAGE_WRITE)
        fs_write.write("image_width", int(self.image_size[1]))
        fs_write.write("image_height", int(self.image_size[0]))
        fs_write.write("camera_name", "rightCamera")
        fs_write.write("camera_matrix", self.right_camera_matrix)
        fs_write.write("distortion_model", "plumb_bob")
        fs_write.write("distortion_coefficients", self.right_distortion)
        fs_write.write("rectification_matrix", self.right_rectification)
        fs_write.write("projection_matrix", self.right_projection)
        fs_write.release()
        print("Calibration save complete")
    
    def load_calibration(self,folderpath):
        print("Loading calibration from folder %s, in files left.yaml and right.yaml...",folderpath)
        fs_read = cv2.FileStorage(folderpath+"/left.yaml", cv2.FILE_STORAGE_READ)
        image_width = int(fs_read.getNode('image_width').real())
        image_height = int(fs_read.getNode('image_height').real())
        self.image_size = (image_height,image_width)
        self.left_camera_matrix = fs_read.getNode('camera_matrix').mat()
        self.left_distortion = fs_read.getNode('distortion_coefficients').mat()
        self.left_rectification = fs_read.getNode('rectification_matrix').mat()
        self.left_projection = fs_read.getNode('projection_matrix').mat()
        fs_read.release()

        fs_read = cv2.FileStorage(folderpath+"/right.yaml", cv2.FILE_STORAGE_READ)
        self.right_camera_matrix = fs_read.getNode('camera_matrix').mat()
        self.right_distortion = fs_read.getNode('distortion_coefficients').mat()
        self.right_rectification = fs_read.getNode('rectification_matrix').mat()
        self.right_projection = fs_read.getNode('projection_matrix').mat()
        fs_read.release()
        self.isReady = True
        print("Calibration load complete")

    def calibrate(self):
        if (self.capture_index > 0):
            # use accumulated object and image points to calibrate the camera
            print("Processing calibration...")
            if (self.capture_index > 30):
                print("Lots of images captured, (suggested to used less than 30) this may take a while...")
            print("Processing first camera calibration...")
            ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(self.objpoints, self.img1points, self.image_size,None,None)
            print("Processing second camera calibration...")
            ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(self.objpoints, self.img2points, self.image_size,None,None)
            
            if (ret1 and ret2):
                print("Processing stereo calibration...")
                (_, _, _, _, _, rotationMatrix, translationVector, _, _) = cv2.stereoCalibrate(
                    self.objpoints, self.img1points, self.img2points,
                    mtx1, dist1,
                    mtx2, dist2,
                    self.image_size, None, None, None, None,
                    cv2.CALIB_FIX_INTRINSIC, self.criteria)

                print("Processing rectification calibration...")
                (leftRectification, rightRectification, leftProjection, rightProjection,
                    dispartityToDepthMap, leftROI, rightROI) = cv2.stereoRectify(
                            mtx1, dist1,
                            mtx2, dist2,
                            self.image_size, rotationMatrix, translationVector,
                            None, None, None, None, None,
                            cv2.CALIB_ZERO_DISPARITY)

                self.left_rectification = leftRectification
                self.right_rectification = rightRectification
                self.left_projection = leftProjection
                self.right_projection = rightProjection
                self.left_distortion = dist1
                self.right_distortion = dist2
                self.left_camera_matrix = mtx1
                self.right_camera_matrix = mtx2

                self.isReady = True

                print("Calibration complete")
                return True
            else:
                print("Calibration failed")
                return False
        else:
            print("Not enough stereo pairs captured")
            return False