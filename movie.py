import scipy
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from processing import process_image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

PROCESS_MOVIE = True

if (PROCESS_MOVIE == False):
    DEBUG = True

def p(img):
    global LEFT_LANE_INFO, RIGHT_LANE_INFO
    result = process_image(img,LEFT_LANE_INFO,RIGHT_LANE_INFO)
    return result

class LaneInfo:

    def __init__(self,thresh=400):
        self.lane_points = None
        self.diffs = None
        self.means = [ ]
        self.accel = np.array([ ])
        self.reject_counts = 0
        self.thresh = thresh
        self.history =  [ ]
        self.max_queue_size = 15

    def push(self,item):
        self.history.append(item)
        if (len(self.history) > self.max_queue_size) :
            self.history = self.history[1:]

    def get_averaged_line(self):
        if (len(self.history) == 0):
            return None
        sm = self.history[0]
        for a in self.history[1:]:
            sm = sm + a

        return sm/len(self.history)

    def filter_lane_points(self,new_lane_points):
        if self.lane_points == None:
            self.lane_points = new_lane_points
            self.push(new_lane_points)
            return (0,0,new_lane_points)

        dist = np.sqrt(np.sum((new_lane_points-self.lane_points)**2,axis=1))

        mean = dist.mean()

        if (len(self.means) == 0):
            accel = mean - 0 
        else:
            accel = mean - self.means[-1]

        if ( np.abs(accel) > self.thresh ) :
            self.reject_counts += 1
            print(accel,mean,self.accel.mean(),"REJECTED",self.reject_counts)
            self.accel = np.append(self.accel, [0])
            self.means = np.append(self.means, [self.means[-1]])
            newline = self.get_averaged_line()
            self.lane_points = newline
            self.push(newline)
            return (accel, mean, newline)
            
        self.accel = np.append(self.accel, [accel])
        self.means = np.append(self.means, [mean])

        if(len(self.accel) > 0):
            print(accel,mean,self.accel.mean())

        self.lane_points = new_lane_points
        self.push(new_lane_points)
        return (accel,mean, new_lane_points)
        # Somehow need to threshold hte lane points.
    

LEFT_LANE_INFO = LaneInfo(650)
RIGHT_LANE_INFO = LaneInfo(500) # 500 is best

if PROCESS_MOVIE == False:
    # inpimg = mpimg.imread('test_images/test1.jpg')
    inpimg = mpimg.imread('test_images/left-lane-at-barrier.jpg')
    inpimg = mpimg.imread('test_images/error_prone_image.jpg')
    inpimg = mpimg.imread('test_images/test1_tree_infront.jpg')

    result = process_image(inpimg,LEFT_LANE_INFO,RIGHT_LANE_INFO)

    plt.title("End result")
    plt.imshow(result)
    plt.show()

else:
    video_output = 'processed_video.mp4'
    clip1 = VideoFileClip("project_video.mp4")

    processed_clip = clip1.fl_image(p)
    processed_clip.write_videofile(video_output, audio=False)

