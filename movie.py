from moviepy.editor import VideoFileClip
from IPython.display import HTML
from processing import process_image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


PROCESS_MOVIE = True

def p(img):
    result = process_image(img)
    return result

if PROCESS_MOVIE == False:
    inpimg = mpimg.imread('test_images/test1.jpg')
    result = process_image(inpimg)

    plt.title("End result")
    plt.imshow(result)
    plt.show()

else:
    video_output = 'processed_video.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    processed_clip = clip1.fl_image(p)
    processed_clip.write_videofile(video_output, audio=False)

