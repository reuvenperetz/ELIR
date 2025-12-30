import cv2
import numpy as np

# Load the 4-channel image
depth_png = cv2.imread('QRISP/ScifiBaseStartStop/270p/DepthMipBiasMinus2/0005/0014.png', cv2.IMREAD_UNCHANGED) # (270, 480, 4)
# 2. Convert to float32 for high-precision math
d = depth_png.astype(np.float32)

# 3. Apply the QRISP weighted sum
# Note: OpenCV loads as BGRA by default, so index 2=R, 1=G, 0=B, 3=A
# If you used a different loader, double check channel order
depth_final = (d[:,:,2] / 255.0 +
               d[:,:,1] / (255.0**2) +
               d[:,:,0] / (255.0**3) +
               d[:,:,3] / (255.0**4))

# 1. Inverse as you did
depth_inv = 1.0 - depth_final

# 2. Apply a power (e.g., 5.0 or 10.0) to "push" mid-tones down
# This makes far things stay black longer and brings out close details
depth_contrasted = np.power(depth_inv, 20.0)

# 3. Clip to be safe and show
depth_vis = np.clip(depth_contrasted * 255, 0, 255).astype('uint8')
cv2.imshow('Better Contrast', depth_vis)

cv2.waitKey(0)