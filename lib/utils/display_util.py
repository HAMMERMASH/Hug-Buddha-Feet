import sys
import time
import cv2
import scipy.misc as misc
import numpy as np
class Processbar():
    process = 0.
    length = 50
    max_process = 0.
    icon = '>'

    def __init__(self,max_process,icon = '>'):
        self.max_process = max_process
        self.process = 0
        self.icon = icon
        assert len(icon) == 1, 'icon should be a char'

    def show_process(self,process = None,finish = False):
        if not process:
            self.process += 1
        else:
            self.process = process
        
        if self.process > self.max_process:
            self.process = self.max_process
        progress = int(self.process*1.0 / self.max_process * self.length)
        bar = '[' + '>' * progress + ' ' * (self.length - progress) + ']' +\
            str(self.process*1.0 / self.max_process * 100)[:4] + '%' + '\r'
        sys.stdout.flush()
        sys.stdout.write(bar)
        
        if finish:
            sys.stdout.write('\n')
            print('Done')
            self.process = 0

#def save_image(image):
    
def show_from_array(image,image_dir = None,name = 'No name'):
    """
    display image from numpy array
    """
    if image_dir != None:
        image = misc.imread(image_dir)
    #cv2 store image in BGR
    #by default in this code it's RGB
    cv_image = np.stack((image[:,:,2],
        image[:,:,1],
        image[:,:,0]),
        axis = 2)
    cv2.imshow(name,cv_image.astype('uint8'))   
    ESC = 27
    while True:
        keycode = cv2.waitKey()
        if keycode != -1: 
            keycode &= 0xFF
            if keycode == ESC:
                break 
    cv2.destroyAllWindows()

def draw_bbox(image,boxes,classes):
    
    boxes = boxes.astype('int')
    for i in range(boxes.shape[0]):
        
        box = boxes[i]
        color = (0,255,0)
        overlay = image.copy()
        # (2) draw shapes:
        cv2.rectangle(overlay, (box[0], box[1]), (box[2],box[3]), color, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = cv2.getTextSize(classes[i],font,0.5,2)
        # print(font_size)
        cv2.rectangle(overlay,(box[0],box[1] - font_size[0][1]),(box[2],box[1]),color,-1)
        #cv2.putText(overlay,'OpenCV',(50,50), font, 0.5,(255,255,255),2,cv2.LINE_AA)
        # (3) blend with the original:
        opacity = 0.5
        cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0, image)
        cv2.putText(image,classes[i],(box[0],box[1]),font,0.5,(255,255,255),1,cv2.LINE_AA)
    return image
    # show_from_array(image)
