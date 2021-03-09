import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import timeit
from scipy.ndimage import gaussian_filter

# start = timeit.default_timer()

def img_to_np_array(path_to_img, dimX, dimY, dimZ):
    f = open(path_to_img, 'rb')
    img_str = f.read()
    img_arr = np.fromstring(img_str, np.float32)
    img = np.reshape(img_arr, (dimX, dimY, dimZ))
    f.close()
    return img

def sum_in_sphere(slice,i_center,j_center,R):
    i_center = int(i_center)
    j_center = int(j_center)
    R = int(R)
    sph_el_sum = 0
    #img = np.copy(slice)
    for i in range(2*R):
        for j in range(2*R):
        # Acceptable voxel distance from sphere center
            d = math.sqrt((i - R) ** 2 + (j - R) ** 2)
            if d < R:
                #print('ok')
                sph_el_sum = sph_el_sum + slice[i_center + i - R, j_center + j - R]
                #img.itemset((i_center + i - R, j_center + j - R), 0.01)
    #plt.imshow(img,cmap='jet')
    #plt.colorbar()
    #plt.clim(0,0.05)
    #plt.show()
    return sph_el_sum


def percent_contrast_background_variability(img, dim, vxl_size, filt_fwhm=0, expand=1, flip=False, plot_slice=False):
    r = 1. / vxl_size # Scaling factor, number of voxels per mm

    # Gaussian filter
    if filt_fwhm != 0:
        sig = r * filt_fwhm / 2.355
        img = gaussian_filter(img, sigma=sig)

    # ROIs are also drawn on slices +/- 1 cm and +/- 2 cm from the central slice
    central_slice_num = int(0.6 * dim)
    slice_numbers = np.array([central_slice_num, central_slice_num - int(10 * r), central_slice_num + int(10 * r),
                              central_slice_num - int(20 * r), central_slice_num + int(20 * r)], np.int)

    # Expand original image, to take into account partial pixels
    vxl_size_expanded = vxl_size / expand
    r_expanded = 1. / vxl_size_expanded

    # Get slices and put them in an array
    slices = []
    for sliceNum in slice_numbers:
        slice = img[sliceNum,:,:]
        slice = np.flip(slice, 0)

        # Some images are flipped after PET image reconstruction
        if flip:
            slice = np.flip(slice, 1)

        slice_expanded = np.repeat(np.repeat(slice, expand, axis=0), expand, axis=1)
        slice_expanded = np.divide(slice_expanded, expand)
        slices.append(slice_expanded)

    slices = np.asanyarray(slices)

    # Plot central slice
    if plot_slice:
        plt_slice = slices[0]
        plt.imshow(plt_slice,cmap='jet')
        plt.colorbar()
        plt.show()

    # Hot spheres, NEMA 2018
    x_sph = np.array([57.2, 28.6, -28.6, -57.2, -28.6, 28.6], np.float32)
    y_sph = np.array([0, -49.54, -49.54, 0, 49.54, 49.54], np.float32)
    R_sph = np.array([18.5, 14, 11, 8.5, 6.5, 5], np.float32)

    i_sph = (dim*expand/2) - np.ceil(y_sph/vxl_size_expanded)
    j_sph = (dim*expand/2) + np.floor(x_sph/vxl_size_expanded)
    R_sph_vxl = np.floor(R_sph/vxl_size_expanded)

    sph_el_sum = np.array([0, 0, 0, 0, 0, 0], np.float32)
    for ii in range(np.size(sph_el_sum)):
        sph_el_sum[ii] = sum_in_sphere(slices[0],i_sph[ii],j_sph[ii],R_sph_vxl[ii])
    #print(sph_el_sum)

    # Background spheres, mm, 12 in each slice, 1.5 cm from the edges and from the hot spheres
    x_b_sph = np.array([-5, 65, -65, 100, -110, 110, -110, 80, -95, -80, 0, -75], np.float32)
    y_b_sph = np.array([-84, -78, -78, -58, -60, -20, -20, 50, 15, 50, 80, -40], np.float32)

    i_b_sph = (dim*expand/2) - np.ceil(y_b_sph/vxl_size_expanded)
    j_b_sph = (dim*expand/2) + np.floor(x_b_sph/vxl_size_expanded)

    # 60 Background spheres,
    background_spheres = np.zeros((60, 6), dtype=float)

    for jj in range(6):
        for kk in range(5):
            for nn in range(12):
                background_spheres[12*kk+nn,jj] = sum_in_sphere(slices[kk],i_b_sph[nn],j_b_sph[nn],R_sph_vxl[jj])
    #print(background_spheres)

    background_spheres_averages = np.mean(background_spheres, axis=0)
    background_spheres_sd = np.std(background_spheres, axis=0)
    percent_contrast = (sph_el_sum / background_spheres_averages - 1) / 3 # C_hot/C_bgr = 4
    percent_background_variability = 100 * (background_spheres_sd / background_spheres_averages)

    return percent_contrast, percent_background_variability



img3D = img_to_np_array('Data/castor_image.img', 120, 120, 120)


percent_contrast, background_variability = percent_contrast_background_variability(img3D, 120,  3, filt_fwhm=5, expand=10, flip=False, plot_slice=True)

print('Percent contrast of hot sphers (37 mm - 10 mm):', percent_contrast)
print('Background variability:', background_variability)

# stop = timeit.default_timer()
# print('Time: ', stop - start)
