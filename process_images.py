#import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import einops as eo

import cv2 as cv

#matplotlib.use('webagg')


#out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
#for _ in range(fps * duration):
    #out.write(data)
def playMat(data, delta=50):
    for k in range(data.shape[-1]):
        frame = data[..., k]
        cv.imshow('frame', frame)
        if cv.waitKey(delta) == ord('q'):
            break
    cv.destroyAllWindows()

def removeFrameColor(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    return gray

def removeDataColor(data, axis):
    new = [removeFrameColor(frame) for frame in np.rollaxis(data, axis)]
    new_arr = eo.rearrange(np.array(new), "t x y -> x y t")
    return new_arr

def highlightLaser(data, axis=0):
    mask = data<np.max(data, axis=axis)
    old_processed_data = np.where(mask, 0, data)
    processed_data = np.zeros(old_processed_data.shape)
    cols = np.arange(old_processed_data.shape[1])[:, np.newaxis]
    rows = np.argmax(old_processed_data != 0, 0)
    processed_data[rows, cols, ...] = old_processed_data[rows, cols, ...]

    
    #processed_data = data<np.max(data, axis=axis)
    print(processed_data.shape)
    return processed_data



def get_zcal_func(pixel_heights, heights):
    x, y = np.asarray(pixel_heights), np.asarray(heights)
    p = np.polyfit(x, y, deg=1)
    print(p[0], p[1])
    return lambda x: p[0]*x + p[1]

def getObjectProjectedHeight(data):
    pixels = np.argwhere(data>0)
    return pixels


def getCalibrationIndex(frame, columns):
    pixels = np.argwhere(frame[:, columns])
    print(pixels)
    _, indecies = np.unique(pixels[:, 1], return_index=True)
    print(indecies)
    new_pixels = pixels[indecies, :]
    return new_pixels[new_pixels[:, 1].argsort(), 0]




if __name__=="__main__":
    mat = sp.io.loadmat('data/Eraser_WebCam_Focus.mat')
    data = next(val for key, val in mat.items() if not key.startswith("__")) 
    decolored_data = removeDataColor(data, -1)
    highlighted_data = highlightLaser(decolored_data)

    processed = highlighted_data
    print(processed.shape)


    calibration_indecies = np.array([0, 100, 400], dtype=np.int32)
    calibration_cols = np.array([0, 2, 7.3])

    cal_heights = getCalibrationIndex(processed[..., 0], calibration_indecies)
    print(cal_heights.shape)
    print(calibration_cols.shape)
    cal_z = get_zcal_func(cal_heights, calibration_cols)

    playMat(processed, delta=1)

    object_projected = getObjectProjectedHeight(processed)
    print(processed.shape)
    print(object_projected.shape)



    ax = plt.axes(projection='3d')

    zdata = cal_z(object_projected[:, 0])
    xdata = object_projected[:, 1] 
    ydata = -object_projected[:, 2] 
    #
    # Data for three-dimensional scattered points
    ax.scatter3D(xdata, ydata, zdata, c=zdata, s=1);
    plt.show()


