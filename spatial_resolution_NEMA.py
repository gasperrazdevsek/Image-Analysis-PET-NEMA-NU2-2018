import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def img_to_np_array(path_to_img, dimX, dimY, dimZ):
    f = open(path_to_img, 'rb')
    img_str = f.read()
    img_arr = np.fromstring(img_str, np.float32)
    img = np.reshape(img_arr, (dimX, dimY, dimZ))
    f.close()
    return img


# NEMA standard - FWHM is determined by linear interpolation between adjacent
# pixels at half the maximum value of the response function.

def fwhm_from_vector(vector, vxl_size=1):
    x = range(np.size(vector))
    ## Find maximum value
    x_poly2 = np.array([np.argmax(vector) - 1, np.argmax(vector), np.argmax(vector) + 1])
    y_poly2 = np.array([vector[np.argmax(vector) - 1], vector[np.argmax(vector)], vector[np.argmax(vector) + 1]])
    coeff_poly2 = np.polyfit(x_poly2, y_poly2, 2)
    poly2 = np.poly1d(coeff_poly2)
    x_max_val = -coeff_poly2[1] / (2 * coeff_poly2[0]) # p = -b/2a
    y_max_val = -(coeff_poly2[1] * coeff_poly2[1] - 4 * coeff_poly2[2] * coeff_poly2[0]) / (4 * coeff_poly2[0]) # q = -D/4a
    # print(x_max_val, y_max_val)
    ## Find FWHM
    for ii in x[:-1]:
        if  vector[ii] < y_max_val/2 and y_max_val/2 < vector[ii + 1]:
            x1 = np.array([ii, ii + 1])
            xp1 = np.linspace(ii, ii + 1, 100)
        if y_max_val/2 < vector[ii]  and y_max_val/2 > vector[ii + 1]:
            x2 = np.array([ii, ii + 1])
            xp2 = np.linspace(ii, ii + 1, 100)
    coeff_line1 = np.polyfit(x1, vector[x1], 1)
    line1 = np.poly1d(coeff_line1)
    coeff_line2 = np.polyfit(x2, vector[x2], 1)
    line2 = np.poly1d(coeff_line2)
    xFWHMl = (y_max_val/2 - coeff_line1[1])/coeff_line1[0]
    xFWHMr = (y_max_val/2 - coeff_line2[1])/coeff_line2[0]
    ## Visualization
    # plt.bar(x, vector)
    # xp = np.linspace(np.argmax(vector) - 1, np.argmax(vector) + 1, 100)
    # plt.plot(xp, poly2(xp), 'g--')
    # plt.plot(xp1, line1(xp1), 'r--')
    # plt.plot(xp2, line2(xp2), 'r--')
    # plt.plot([xFWHMl, xFWHMr], [y_max_val/2, y_max_val/2], 'r', linewidth=3.0)
    # plt.show()
    fwhm = (xFWHMr - xFWHMl) * vxl_size
    # print(fwhm)
    return fwhm

