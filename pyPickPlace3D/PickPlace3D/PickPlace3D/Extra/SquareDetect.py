import cv2
import numpy as np

class DetectObject():
    def __init__(self,x_real,y_real,z_real,x_pix,y_pix,angle,contour):
        self.x_real = x_real
        self.x_pix = x_pix
        self.y_real = y_real
        self.y_pix = y_pix
        self.z_real = z_real
        self.angle = angle
        self.contour = contour

class SquareDetect():
    def __init__(self,pixel_size=0.00000345,focal_length=0.008,baseline=0.3,binning=1.0,fx=582.3034,isShowThreshold=False):
        self.isShowThreshold = isShowThreshold
        self.pixel_size = pixel_size
        self.focal_length = focal_length
        #self.fx = 592.161956
        #self.fx = 1164.6068743790038
        self.fx = fx
        self.baseline = baseline
        self.binning = binning

    def pixel_coords_to_meters(self,x_pix,y_pix,z_real):
        # Convert from pixel number to real world distance. 
        # Fomula used is X (world) = Z(world) x ( x(pixel) / focal lenth )

        x_real = z_real * ((x_pix * self.pixel_size)/self.focal_length)
        y_real = z_real * ((y_pix * self.pixel_size)/self.focal_length)
        return x_real,y_real

    def disp_to_depth(self,disp):
        return self.baseline * self.fx / disp

    def prep_image(self,img,thresh_min=70,thresh_max=255):
        # prepare image for contour detection
        # blur image
        img_blur = cv2.blur(img,(5,5))
        # threshold image to binary image
        min_threshold = thresh_min #90
        ret, img_thresh = cv2.threshold(img_blur, min_threshold, thresh_max, cv2.THRESH_BINARY)
        # close holes in binary image
        #kernel = np.ones((5,5),np.uint8)
        #img_close_holes = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)

        if self.isShowThreshold:
            cv2.imshow("threshold_pre_fill",img_thresh)

        # Copy the thresholded image.
        im_floodfill = img_thresh.copy()
        
        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = img_thresh.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        
        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0,0), 255)
        
        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        
        # Combine the two images to get the foreground.
        img_close_holes = img_thresh | im_floodfill_inv
        return img_thresh

    def get_contour_shape(self,contour):
        shape = "unidentified"
        peri = cv2.arcLength(contour, True)
        roundness = 0.03 #0.02
        approx = cv2.approxPolyDP(contour, roundness * peri, True)

		# if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"

		# if the shape has 4 vertices, it is either a square or
		# a rectangle
        elif len(approx) == 4:
			# compute the bounding box of the contour and use the
			# bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)

			# a square will have an aspect ratio that is approximately
			# equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.85 and ar <= 1.2 else "rectangle"

		# if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"

		# otherwise, we assume the shape is a circle
        else:
            shape = "circle"

		# return the name of the shape
        return shape, approx

    def find_contours(self,img,isFilterSize=True):
        # find contours in binary image
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if (len(contours) == 0):
            print("No contours found")
            return False,None
        else:
            if (isFilterSize):
                contours_filtered = []
                for contour in contours:
                    # get contour area
                    contour_area = cv2.contourArea(contour)
                    # get total area of image
                    image_area = img.shape[0] * img.shape[1]
                    # contour area must be at least 0.05% of the image area
                    if (contour_area > (image_area*0.005)):
                        # rectangle area must be less than 95% of the image
                        if (contour_area < (image_area*0.95)):
                            contours_filtered.append(contour)
                        else:
                            pass
                            #print("Contour too large")
                    else:
                        pass
                        #print("Contour too small")
                if (len(contours_filtered) == 0):
                    print("Not contours found after filtering size")
                    return False,None
                else:
                    return True,contours_filtered
            else:
                return True,contours

    def find_square_contours(self,img):
        # find contours in binary image
        res, contours = self.find_contours(img)
        if (res):
            # find square contours
            square_contours = []
            for contour in contours:
                # check contour is square or rectangle
                shape_name,approx = self.get_contour_shape(contour)
                if shape_name == "square" or shape_name == "rectangle":
                    square_contours.append(contour)
                else:
                    pass
                    #print("Shape is not square or rectangle: {}".format(shape_name))
            if (len(square_contours) > 0):
                return True, square_contours
            else:
                print("Failed to find any square contours")
                return False, None
        else:
            print("No contours found")
            return False,None

    def find_largest_contour(self,img):
        # find contours in binary image
        res, contours = self.find_contours(img)
        if (res):
            # find contour with largest area
            largest_contour = max(contours, key = cv2.contourArea)
            return True,largest_contour
        else:
            print("No contours found")
            return False,None

    def get_coord_from_contour(self,contour):
        # fit rectangle to contour
        contour_rect = cv2.minAreaRect(contour)

        # get rectangle angle
        angle = contour_rect[2]
        if (contour_rect[1][0] < contour_rect[1][1]):
            angle = angle - 90

        # get rectangle center
        x_pix = contour_rect[0][0]
        y_pix = contour_rect[0][1]

        return x_pix, y_pix, angle

    def crop_to_contour(self,image,contour):
        # crop image to only show the detected contour for comparision with right image
        mask = np.zeros_like(image) # Create mask where white is what we want, black otherwise
        cv2.drawContours(mask, [contour], 0, 255, -1) # Draw filled contour in mask
        out_image = np.zeros_like(image) # Extract out the object and place into output image
        out_image[mask == 255] = image[mask == 255]

        # Now crop
        (y, x) = np.where(mask == 255)
        (topy, topx) = (np.min(y), np.min(x))
        (bottomy, bottomx) = (np.max(y), np.max(x))
        out_image = out_image[topy:bottomy+1, topx:bottomx+1]
        return out_image

    def detect_squares(self,image,ROI=[586,800,545,1500],fixed_z=1.0):
        # Detect square objects in image
        # Currently z is fixed (default=1m)
        # TODO create Z plane that returns the z value based on measurement
        # TODO get z from 3D trianglulation

        crop_image = image[ROI[0]:ROI[0]+ROI[1],ROI[2]:ROI[2]+ROI[3]]

        out_image = crop_image

        # prepare image for contour detection (convert to binary)
        img_proc = self.prep_image(crop_image)

        if self.isShowThreshold:
            cv2.imshow("threshold",img_proc)

        ret, square_contours = self.find_square_contours(img_proc)

        if (not ret):
            print("Failed to find contour")
            return False,None,out_image
        
        squares = []
        i = 0
        for contour in square_contours:
            # draw contour on original image
            contourColor = (0,25,0)
            cv2.drawContours(crop_image, [contour], 0, contourColor, 3)
            out_image = crop_image

            

            object_crop_image = self.crop_to_contour(out_image,contour)
            #cv2.imshow("object {}".format(i),object_crop_image)

            # Only only image is being used to fix z to defined value (m)
            z = fixed_z

            x_pix, y_pix, angle = self.get_coord_from_contour(contour)

            x_pix_s = x_pix * self.binning
            y_pix_s = y_pix * self.binning

            out_image = cv2.drawMarker(out_image,(int(x_pix),int(y_pix)),0,cv2.MARKER_CROSS)
            
            # convert pixel co-ordinates to real world (meters)
            x_real,y_real = self.pixel_coords_to_meters(x_pix_s,y_pix_s,z)

            sq = DetectObject(x_real,y_real,z,x_pix,y_pix,angle,contour)

            squares.append(sq)
            i+=1

        return True,squares,out_image

    def detect_largest_contour(self,image,ROI=[586,800,529,1500],fixed_z=1.0,image2=None,ROI2=[621,800,615,1500]):
        # Detect largest contour in image
        # If using dual images will average result from both images
        # Currently z is fixed (default=1m)

        # crop image for world zero to match image zero (top left)
        #crop_image_l = image[614:1500,497:2000] #TIS camera offset
        #crop_image_l = image[696:1500,569:2000] #Basler camera offset

        image_size_orig = (image.shape[0]*self.binning,image.shape[1]*self.binning)
        image_size_bin = (image.shape[0],image.shape[1])
        ROI_bin = [0,0,0,0]
        ROI2_bin = [0,0,0,0]
        ROI_bin[0] = int(ROI[0] / self.binning)
        ROI_bin[1] = int(ROI[1] / self.binning)
        ROI_bin[2] = int(ROI[2] / self.binning)
        ROI_bin[3] = int(ROI[3] / self.binning)
        ROI2_bin[0] = int(ROI2[0] / self.binning)
        ROI2_bin[1] = int(ROI2[1] / self.binning)
        ROI2_bin[2] = int(ROI2[2] / self.binning)
        ROI2_bin[3] = int(ROI2[3] / self.binning)

        crop_image = image[ROI_bin[0]:ROI_bin[0]+ROI_bin[1],ROI_bin[2]:ROI_bin[2]+ROI_bin[3]]

        out_image = crop_image
        
        if (not image2 is None):
            crop_image2 = image2[ROI2_bin[0]:ROI2_bin[0]+ROI2_bin[1],ROI2_bin[2]:ROI2_bin[2]+ROI2_bin[3]]
            out_image = cv2.hconcat([crop_image, crop_image2])


        # prepare image for contour detection (convert to binary)
        img_proc = self.prep_image(crop_image)
        if (not image2 is None):
            img_proc2 = self.prep_image(crop_image2)

        if self.isShowThreshold:
            cv2.imshow("threshold",img_proc)
            if (not image2 is None):
                cv2.imshow("threshold2",img_proc2)

        ret, largest_contour = self.find_largest_contour(img_proc)
        if (not image2 is None):
            ret2, largest_contour2 = self.find_largest_contour(img_proc2)

        if (not ret):
            print("Failed to find contour")
            return False,None,None,None,None,out_image

        if ((not image2 is None) and (not ret2)):
            print("Failed to find contour in second image")
            return False,None,None,None,None,out_image
        
        # draw contour on original image
        contourColor = (0,25,0)
        cv2.drawContours(crop_image, [largest_contour], 0, contourColor, 3)
        out_image = crop_image
        if (not image2 is None):
            cv2.drawContours(crop_image2, [largest_contour2], 0, contourColor, 3)
            out_image = cv2.hconcat([crop_image, crop_image2])

        x_pix, y_pix, angle = self.get_coord_from_contour(largest_contour)
        if (not image2 is None):
            x_pix2, y_pix2, angle2 = self.get_coord_from_contour(largest_contour2)

        # undo binning
        x_pix = x_pix * self.binning
        y_pix = y_pix * self.binning
        if (not image2 is None):
            x_pix2 = x_pix2 * self.binning
            y_pix2 = y_pix2 * self.binning
        
        if (image2 is None):
            # Only only image is being used to fix z to defined value (m)
            z = fixed_z
            # convert pixel co-ordinates to real world (meters)
            x_real,y_real = self.pixel_coords_to_meters(x_pix,y_pix,z)
        else:
            #M = cv2.moments(cnt_l)
            #box_fit_l = cv2.minAreaRect(cnt_l)
            #box_points_l = cv2.boxPoints(rect_l)
            #box_l = np.int0(box_points_l)
            # TODO get x and y of contour using template matching
            # TODO triangulate co-ordinates in both images to find z
            
            z = 1.0
            x_real,y_real = self.pixel_coords_to_meters(x_pix,y_pix,z)
            x_real2,y_real2 = self.pixel_coords_to_meters(x_pix2,y_pix2,z)

            x_pix_unroi = x_pix + ROI[2]
            y_pix_unroi = y_pix + ROI[0]

            x_pix2_unroi = x_pix2 + ROI2[2]
            y_pix2_unroi = y_pix2 + ROI2[0]

            disp_x = (x_pix_unroi-x_pix2_unroi)
            disp_y = (y_pix_unroi-y_pix2_unroi)

            d = self.disp_to_depth(disp_x)

            print("Disp: {}, {}".format(disp_x,disp_y))
            print("Z: {}".format(d))

            print("Cam 1: {},{} @ {}".format(x_real,y_real,angle))
            print("Cam 2: {},{} @ {}".format(x_real2,y_real2,angle2))
            

            x_real = (x_real + x_real2) / 2
            y_real = (y_real + y_real2) / 2
            angle = (angle + angle2) / 2

        return True,x_real,y_real,z,angle,out_image