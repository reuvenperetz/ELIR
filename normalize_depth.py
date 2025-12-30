import cv2
import numpy as np

# Load the 4-channel image
depth_png = cv2.imread('QRISP/TestSet/AbandonedSchool/540p/DepthMipBiasMinus1/0000/0040.png', cv2.IMREAD_UNCHANGED) # (270, 480, 4)
# 2. Convert to float32 for high-precision math
d = depth_png.astype(np.float32)

# 3. Apply the QRISP weighted sum
# Note: OpenCV loads as BGRA by default, so index 2=R, 1=G, 0=B, 3=A
# If you used a different loader, double check channel order
depth_final = (d[:,:,2] / 255.0 +
               d[:,:,1] / (255.0**2) +
               d[:,:,0] / (255.0**3) +
               d[:,:,3] / (255.0**4))

# 4. Clean up the view (Remove "Inf" and Normalize)
depth_vis = cv2.normalize(depth_final, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

cv2.imshow('Silky Smooth Depth', depth_vis)
cv2.waitKey(0)