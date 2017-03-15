# Compute the camera calibration matrix

# Go through each image in the camera calibration image
import scipy
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
import matplotlib.image as mpimg

# Arrays to store object points and image points from all the images

DEBUG = False



def find_obj_img_points():
    objp = np.zeros((6*9,3),np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
    
    objpoints = [ ]
    imgpoints = [ ]
    images = glob.glob('camera_cal/*.jpg')

    for idx,fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        
        if ret == 1:
            objpoints.append(objp)
            imgpoints.append(corners)

            cv2.drawChessboardCorners(img, (9,6), corners, ret)
            # cv2.imshow('img',img)
            # cv2.waitKey(500)

    # cv2.destoryAllWindows()
    return objpoints, imgpoints

CROP_Y = 410

def get_transformed(img, img_size):
    global DEBUG
    
    perspective = np.array([[294,665],[1014,665],[710,465],[576,465]],dtype="float32")
    noperspective = np.array([[400,665],[900,665],[900,465],[400,465]],dtype="float32")

    perspective = np.array([[294,665-CROP_Y],[1014,665-CROP_Y],[710,465-CROP_Y],[576,465-CROP_Y]],dtype="float32")
    noperspective = np.array([[400,665-CROP_Y],[900,665-CROP_Y],[900,465-CROP_Y],[400,465-CROP_Y]],dtype="float32")

    perspective = np.array([[294,665-CROP_Y],[1014,665-CROP_Y],[710,465-CROP_Y],[576,465-CROP_Y]],dtype="float32")
    noperspective = np.array([[400,665-CROP_Y],[1100,665-CROP_Y],[1100,465-CROP_Y],[400,465-CROP_Y]],dtype="float32")

    if DEBUG:
        print(perspective.shape)
        print(noperspective.shape)
    
    M = cv2.getPerspectiveTransform(perspective, noperspective)

    Minv = cv2.getPerspectiveTransform(noperspective, perspective)
    
    warped = cv2.warpPerspective(img,M,img_size,flags=cv2.INTER_LINEAR )
    return M,warped,Minv


def crop_y(img, from_top, from_bottom):
    y = img.shape[0]
    x = img.shape[1]
    return img[from_top:(y-from_bottom),:,:]


def hsv_v_mask(img,gray):
    
    hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    hsv_h_channel = hsv[:,:,0]
    hsv_s_channel = hsv[:,:,1]
    hsv_v_channel = hsv[:,:,2]

    t = [hsv_h_channel,hsv_s_channel,hsv_v_channel] 
    v_thresh=[30,180]

    for i in range(3):
        vbinary = np.zeros_like(gray)
        v = t[i]
        vbinary[ (v >= v_thresh[0]) & (v <= v_thresh[1]) ] = 1
    
        if DEBUG:
            plt.title("HSV -V {}".format(i))
            plt.imshow(vbinary)
            plt.show()

    return vbinary
    

# def pipeline(img, s_thresh=(120,150), sx_thresh=(50,120)):
# def pipeline(img, s_thresh=(50,225), sx_thresh=(20,120)): CORRECT
# s_thresh=(170, 255), sx_thresh=(20, 100)
def pipeline(img, s_thresh=(50,255), sx_thresh=(20,120)):
    global DEBUG
    
    img = np.copy(img)
    y = img.shape[0]
    x = img.shape[1]
    
    img = crop_y(img, 410, y-670)

    if DEBUG:
        plt.imshow(img)
        plt.show()
        print(" X {}  Y {} shape:{}".format(img.shape[1],img.shape[0],img.shape))
    
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]


    # Sobel X
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray,cv2.CV_64F, 0,1)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    if DEBUG:
        plt.title("Scaled Sobel")
        plt.imshow(scaled_sobel)
        plt.show()

    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[ (scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    if DEBUG:
        plt.title("Sobel Gradient")
        plt.imshow(sxbinary)
        plt.show()
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    if DEBUG:
        plt.title("S BINARY")
        plt.imshow(s_binary)
        plt.show()
    
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))
 
    combined_binary = np.zeros_like(sxbinary)
    
    combined_binary[ (s_binary==1) | (sxbinary == 1)  ] = 1

    if DEBUG:
        plt.title("COMBINED BINARY")
        plt.imshow(combined_binary)
        plt.show()
        cv2.imwrite("test_images/test1_combined.jpg",combined_binary)
            
    M , warped, Minv = get_transformed(combined_binary, (combined_binary.shape[1], combined_binary.shape[0]))

    if DEBUG:
        cv2.imwrite("test_images/test1_warped.jpg",warped)
    
    return warped, Minv
    # return color_binary

    # apply the transform

    
def fit_poly_to_lane_line(binary_warped):
    global first_image, DEBUG

    if DEBUG:
        print("Binary Warped {} ".format(binary_warped.shape))
        print("{}".format(np.unique(binary_warped)))
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image

    if first_image: 

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        first_image = False
    else:
        # Assume you now have a new warped binary image 
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)


    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    if DEBUG:
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()
        cv2.imwrite("test_images/test1_color_fit_lines.jpg",out_img)
    
    return [left_fitx,right_fitx,ploty ]


def get_curvature(left_fitx, right_fitx,ploty):
    # Generate some fake data to represent lane-line pixels

    ploty = np.linspace(0, 259, num=260)# to cover same y-range as image

    quadratic_coeff = 3e-4 # arbitrary quadratic coefficient

    # For each y position generate random x position within +/-50 pix
    # of the line base position in each case (x=200 for left, and x=900 for right)
    leftx = np.array([200 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) for y in ploty])
    rightx = np.array([900 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) for y in ploty])

    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y


    # Fit a second order polynomial to pixel positions in each fake lane line
    left_fit = np.polyfit(ploty, leftx, 2)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fit = np.polyfit(ploty, rightx, 2)
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    if DEBUG:
        # Plot up the fake data
        mark_size = 3
        plt.plot(leftx, ploty, 'o', color='red', markersize=mark_size)
        plt.plot(rightx, ploty, 'o', color='blue', markersize=mark_size)
        plt.xlim(0, 1280)
        plt.ylim(0, 720)
        plt.plot(left_fitx, ploty, color='green', linewidth=3)
        plt.plot(right_fitx, ploty, color='green', linewidth=3)
        plt.gca().invert_yaxis() # to visualize as we do the images
        plt.show()

    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

    if DEBUG:
        print(left_curverad, right_curverad)

    mid_point_pixel_space = (rightx[0] - leftx[0])/2 + leftx[0]
        
    ym_per_pix = 30/260 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters

    
    mid_screen = 1280/2
    offset_px = mid_screen-mid_point_pixel_space
    offset = np.abs((mid_screen - mid_point_pixel_space)*xm_per_pix)

    if DEBUG:
        print(left_curverad, 'm', right_curverad, 'm')
    
    return (left_curverad,right_curverad,offset)
    
def get_dist_mtx(img):
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    return (mtx,dist)


def draw_poly(img, vertices, color=[0,255,0],thickness=2):
    for i in range(len(vertices)):
        if (i == (len(vertices) - 1)):
            cv2.line(img,vertices[i],vertices[0],color,thickness)
        else:
            cv2.line(img,vertices[i], vertices[i+1], color, thickness)

def find_transform_points():
    # Read in the straight line image
    img = mpimg.imread('test_images/straight_lines1.jpg')

    draw_poly(img,[(294,665),(1014,665),(710,465),(576,465)])
    # 925 and 300

    draw_poly(img,[(400,665),(900,665),(900,465),(400,465)],color=[255,0,0])
    plt.imshow(img)
    plt.show()


def draw_wrap_image(undist, warped,ploty,left_fitx,right_fitx,Minv,left_curve_rad,right_curve_rad,offset,left_lane_info=None,right_lane_info=None):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])

    accel_l,mean_l,pts_left_to_use = left_lane_info.filter_lane_points(pts_left)
    accel_r,mean_r,pts_right_to_use = right_lane_info.filter_lane_points(pts_right)

    pts = np.hstack((pts_left_to_use, pts_right_to_use))

    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Draw the lane lines
    cv2.polylines(color_warp, np.int_([pts_right_to_use]), False ,(255,0, 0),thickness=30)
    cv2.polylines(color_warp, np.int_([pts_left_to_use]), False,(0,0,255),thickness=30)



    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    # image
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))

    rows = newwarp.shape[0]
    cols = newwarp.shape[1]
    
    # Move the newwarp down on my non cropped image
    M= np.float32([[1,0,0],[0,1,410]])

    newwarp = cv2.warpAffine(newwarp,M, (cols,rows))
    
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    cv2.putText(result,"R: {:6.2f}m Off:{:4.2f}m Lu:{:6.2f} La:{:6.2f} Ru:{:6.2f} Ra:{:6.2f}".format((left_curve_rad+right_curve_rad)/2,offset,mean_l,accel_l,mean_r,accel_r ), (10,100), cv2.FONT_HERSHEY_SIMPLEX,1, 255)
    
    if DEBUG:
        plt.imshow(result)
        plt.show()
        cv2.imwrite("test_images/test1_result.jpg",result)
    
    return result


# find_transform_points()
# if 1:
#    exit()


def load_from_pickle():
    dist = pickle.load( open("./dist_pickle.p", "rb") )
    return dist["mtx"], dist["dist"]


def save_pickle(mtx,dist):
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open("./dist_pickle.p","wb"))



def process_image(img,left_lane_info=None,right_lane_info=None):
    global first_image
    no_file = False
    mtx = None
    dist = None 

    try:
        mtx, dist = load_from_pickle()
    except:
        no_file = True

    if no_file:
        objpoints, imgpoints = find_obj_img_points()
        mtx, dist = get_dist_mtx(img)
        save_pickle(mtx,dist)

    dst = cv2.undistort(img, mtx, dist, None,mtx)
    # cv2.imwrite('test_images/test1_unsdist.jpg',dst)


    if DEBUG:
        test_img = mpimg.imread("test_images/straight_lines1.jpg")
        y = test_img.shape[0]
        test_img =  crop_y(test_img, 410, y-670)    
        pM, pers, pMinv = get_transformed(test_img,(test_img.shape[1],test_img.shape[0]))

    
    if DEBUG and False:
        fig = plt.figure(1,figsize=(8,10))
        plt.title('Perspective Transform')
        plt.imshow(test_img)
        plt.show()
        plt.imshow(pers)
        plt.show()

    

    th, Minv = pipeline(dst)

    if DEBUG:
        plt.title("TH pipeline output")
        cv2.imwrite("test_images/test1_combined_thresholdx.jpg",th)
        plt.imshow(th)
        plt.show()
    
    
    first_image = True
    left_fitx, right_fitx, ploty = fit_poly_to_lane_line(th)

    if DEBUG:
        print("left_fitx:",left_fitx)
        print("right_fitx:", right_fitx)
        print("ploty :", ploty)
    
    left_curverad, right_curverad, offset = get_curvature(left_fitx, right_fitx,ploty)
    result = draw_wrap_image(dst , th,ploty,left_fitx,right_fitx,Minv,left_curverad,right_curverad,offset,left_lane_info,right_lane_info)

    if DEBUG and False :
        f, ((ax1 , ax2) ,(ax3 , ax4)) = plt.subplots(2,2,figsize=(20,10))
        ax1.imshow(img)
        ax1.set_title('Original Image' , fontsize = 30)
        ax2.imshow(dst)
        ax2.set_title('Undistorted Image', fontsize = 30)
        ax3.imshow(th)
        ax3.set_title('Mask', fontsize = 30)
        f.show()

    save_pickle(mtx,dist)

    if DEBUG:
        plt.show()

    # cv2.imwrite("test_images/test1_threshold.jpg",th)

    return result


    
