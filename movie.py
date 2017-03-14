import scipy
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from processing import process_image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

PROCESS_MOVIE = True


def p(img):
    global LEFT_LANE_INFO, RIGHT_LANE_INFO
    result = process_image(img,LEFT_LANE_INFO,RIGHT_LANE_INFO)
    return result

class LaneInfo:

    def __init__(self):
        self.lane_points = None
        self.diffs = None
        self.means = [ ]
        self.accel = np.array([ ])
        self.reject_counts = 0

    def filter_lane_points(self,new_lane_points):
        if self.lane_points == None:
            self.lane_points = new_lane_points
            return (0,0,new_lane_points)

        dist = np.sqrt(np.sum((new_lane_points-self.lane_points)**2,axis=1))

        mean = dist.mean()

        if (len(self.means) == 0):
            accel = mean - 0 
        else:
            accel = mean - self.means[-1]

        if ( np.abs(accel) > 400 ) :
            self.reject_counts += 1
            print(accel,mean,self.accel.mean(),"REJECTED",self.reject_counts)
            self.accel = np.append(self.accel, [0])
            self.means = np.append(self.means, [self.means[-1]])
            return (accel, mean, self.lane_points)
            
        self.accel = np.append(self.accel, [accel])
        self.means = np.append(self.means, [mean])

        if(len(self.accel) > 0):
            print(accel,mean,self.accel.mean())

        self.left_points = new_lane_points
        return (accel,mean, new_lane_points)
        # Somehow need to threshold hte lane points.
    
        

if PROCESS_MOVIE == False:
    # inpimg = mpimg.imread('test_images/test1.jpg')
    inpimg = mpimg.imread('test_images/error_prone_image.jpg')
    inpimg = mpimg.imread('test_images/left-lane-at-barrier.jpg')
    LEFT_LANE_INFO = LaneInfo()
    RIGHT_LANE_INFO = LaneInfo()
    result = process_image(inpimg,LEFT_LANE_INFO,RIGHT_LANE_INFO)

    plt.title("End result")
    plt.imshow(result)
    plt.show()

else:
    video_output = 'processed_video.mp4'
    clip1 = VideoFileClip("project_video.mp4")

    LEFT_LANE_INFO = LaneInfo()
    RIGHT_LANE_INFO = LaneInfo()
    processed_clip = clip1.fl_image(p)
    processed_clip.write_videofile(video_output, audio=False)

