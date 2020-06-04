import cv2
import numpy as np
from matplotlib import pyplot as plt

# create background for beams
background = np.zeros([200, 200])

# size of kernel for beam 1
kernel_size_1 = 50
sigma_1 = 15

# size of kernel for beam 2
kernel_size_2 = 50
sigma_2 = 15

# generate Gaussian kernel 1 with parameters
kernel_size_1 = kernel_size_1 * 2
gauss_kernel_1 = cv2.getGaussianKernel(kernel_size_1, sigma_1)
gauss_kernel_1 = gauss_kernel_1 * gauss_kernel_1.T

# generate Gaussian kernel 2 with parameters
kernel_size_2 = kernel_size_2 * 2
gauss_kernel_2 = cv2.getGaussianKernel(kernel_size_2, sigma_2)
gauss_kernel_2 = gauss_kernel_2 * gauss_kernel_2.T

# merge kernels with background
background[:100, 100:] = gauss_kernel_1
background[100:, :100] = gauss_kernel_2


fig = plt.figure(frameon=False)
fig.set_size_inches(5, 5)

ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

plt.imshow(background)

# plt.title('Gaussian beam')

# ax = plt.gca()
# ax.spines['left'].set_position('center')
# ax.spines['bottom'].set_position('center')
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)


plt.savefig("source.png",)
# plt.show()
