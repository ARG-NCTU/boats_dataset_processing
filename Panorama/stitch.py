import cv2
import numpy as np
import features


# def blendingMask(height, width, barrier, smoothing_window, left_biased=True):
#     assert barrier < width
#     mask = np.zeros((height, width))

#     offset = int(smoothing_window / 2)
#     try:
#         if left_biased:
#             mask[:, barrier - offset : barrier + offset + 1] = np.tile(
#                 np.linspace(1, 0, 2 * offset + 1).T, (height, 1)
#             )
#             mask[:, : barrier - offset] = 1
#         else:
#             mask[:, barrier - offset : barrier + offset + 1] = np.tile(
#                 np.linspace(0, 1, 2 * offset + 1).T, (height, 1)
#             )
#             mask[:, barrier + offset :] = 1
#     except BaseException:
#         if left_biased:
#             mask[:, barrier - offset : barrier + offset + 1] = np.tile(
#                 np.linspace(1, 0, 2 * offset).T, (height, 1)
#             )
#             mask[:, : barrier - offset] = 1
#         else:
#             mask[:, barrier - offset : barrier + offset + 1] = np.tile(
#                 np.linspace(0, 1, 2 * offset).T, (height, 1)
#             )
#             mask[:, barrier + offset :] = 1

#     return cv2.merge([mask, mask, mask])

def blendingMask(height, width, barrier, smoothing_window, left_biased=True):
    assert barrier < width
    mask = np.zeros((height, width), dtype=np.float32)

    # Adjust the offset to control the transition region
    offset = int(smoothing_window / 2)
    
    # Create a smoother gradient using a cosine function for transition
    # You can adjust the frequency of the cosine to modify steepness
    x = np.linspace(0, np.pi, 2 * offset + 1)  # Adjust the spread by changing the np.pi range
    cosine_gradient = 0.5 * (1 + np.cos(x))  # Cosine transition from 1 to 0

    if left_biased:
        mask[:, barrier - offset : barrier + offset + 1] = np.tile(
            cosine_gradient, (height, 1)
        )
        mask[:, : barrier - offset] = 1
    else:
        mask[:, barrier - offset : barrier + offset + 1] = np.tile(
            cosine_gradient[::-1], (height, 1)
        )
        mask[:, barrier + offset :] = 1

    return cv2.merge([mask, mask, mask])

def panoramaBlending(dst_img_rz, src_img_warped, width_dst, side, showstep=False):
    """Given two aligned images @dst_img and @src_img_warped, and the @width_dst is width of dst_img
    before resize, that indicates where there is the discontinuity between the images,
    this function produce a smoothed transient in the overlapping.
    @smoothing_window is a parameter that determines the width of the transient
    left_biased is a flag that determines whether it is masked the left image,
    or the right one"""

    h, w, _ = dst_img_rz.shape
    smoothing_window = int(width_dst / 8)
    barrier = width_dst - int(smoothing_window / 2)
    mask1 = blendingMask(
        h, w, barrier, smoothing_window=smoothing_window, left_biased=True
    )
    mask2 = blendingMask(
        h, w, barrier, smoothing_window=smoothing_window, left_biased=False
    )

    if showstep:
        nonblend = src_img_warped + dst_img_rz
    else:
        nonblend = None
        leftside = None
        rightside = None

    if side == "left":
        dst_img_rz = cv2.flip(dst_img_rz, 1)
        src_img_warped = cv2.flip(src_img_warped, 1)
        dst_img_rz = dst_img_rz * mask1
        src_img_warped = src_img_warped * mask2
        pano = src_img_warped + dst_img_rz
        pano = cv2.flip(pano, 1)
        if showstep:
            leftside = cv2.flip(src_img_warped, 1)
            rightside = cv2.flip(dst_img_rz, 1)
    else:
        dst_img_rz = dst_img_rz * mask1
        src_img_warped = src_img_warped * mask2
        pano = src_img_warped + dst_img_rz
        if showstep:
            leftside = dst_img_rz
            rightside = src_img_warped

    return pano, nonblend, leftside, rightside


def warpTwoImages(src_img, dst_img, iteration, H_given, showstep=False):

    # generate Homography matrix
    if iteration == 0:
            H, _ = features.generateHomography(src_img, dst_img)
    else:
        H = H_given

    # get height and width of two images
    height_src, width_src = src_img.shape[:2]
    height_dst, width_dst = dst_img.shape[:2]

    # extract conners of two images: top-left, bottom-left, bottom-right, top-right
    pts1 = np.float32(
        [[0, 0], [0, height_src], [width_src, height_src], [width_src, 0]]
    ).reshape(-1, 1, 2)
    pts2 = np.float32(
        [[0, 0], [0, height_dst], [width_dst, height_dst], [width_dst, 0]]
    ).reshape(-1, 1, 2)

    #print(H)
    try:
        # aply homography to conners of src_img
        pts1_ = cv2.perspectiveTransform(pts1, H)
        pts = np.concatenate((pts1_, pts2), axis=0)

        # find max min of x,y coordinate
        [xmin, ymin] = np.int64(pts.min(axis=0).ravel() - 0.5)
        [_, ymax] = np.int64(pts.max(axis=0).ravel() + 0.5)
        t = [-xmin, -ymin]

        # top left point of image which apply homography matrix, which has x coordinate < 0, has side=left
        # otherwise side=right
        # source image is merged to the left side or right side of destination image
        if pts[0][0][0] < 0:
            side = "left"
            width_pano = width_dst + t[0]
        else:
            width_pano = int(pts1_[3][0][0])
            side = "right"
        height_pano = ymax - ymin

        # Translation
        # https://stackoverflow.com/a/20355545
        Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
        src_img_warped = cv2.warpPerspective(
            src_img, Ht.dot(H), (width_pano, height_pano)
        )
        # generating size of dst_img_rz which has the same size as src_img_warped
        dst_img_rz = np.zeros((height_pano, width_pano, 3))
        if side == "left":
            dst_img_rz[t[1] : height_src + t[1], t[0] : width_dst + t[0]] = dst_img
        else:
            dst_img_rz[t[1] : height_src + t[1], :width_dst] = dst_img

        # blending panorama
        pano, nonblend, leftside, rightside = panoramaBlending(
            dst_img_rz, src_img_warped, width_dst, side, showstep=showstep
        )

        # croping black region
        pano = crop(pano, height_dst, pts)
        return pano, nonblend, leftside, rightside
    except BaseException:
        raise Exception("Please try again with another image set!")


def multiStitching(list_images, iteration):
    """assume that the list_images was supplied in left-to-right order, choose middle image then
    divide the array into 2 sub-arrays, left-array and right-array. Stiching middle image with each
    image in 2 sub-arrays. @param list_images is The list which containing images, @param smoothing_window is
    the value of smoothy side after stitched, @param output is the folder which containing stitched image
    """
    n = int(len(list_images) / 2 + 0.5)
    left = list_images[:n]
    right = list_images[n - 1 :]
    right.reverse()
    while len(left) > 1:
        dst_img = left.pop()
        src_img = left.pop()
        H = np.array([[ 3.10108109e+00,  2.10948131e-01, -1.46998946e+03],
                      [ 6.10444571e-01,  2.94902407e+00, -3.80442798e+02],
                      [ 3.05118266e-03,  6.78256390e-04,  1.00000000e+00]])
        left_pano, _, _, _ = warpTwoImages(src_img, dst_img, iteration, H)
        left_pano = left_pano.astype("uint8")
        left.append(left_pano)
    while len(right) > 1:
        dst_img = right.pop()
        src_img = right.pop()
        H = np.array([[ 4.42896384e-01, -3.22459494e-02,  4.48900445e+02],
                      [-1.87434195e-01,  9.06295667e-01,  4.59021205e+00],
                      [-8.66424831e-04, -1.85663837e-05,  1.00000000e+00]])
        right_pano, _, _, _ = warpTwoImages(src_img, dst_img, iteration, H)
        right_pano = right_pano.astype("uint8")
        right.append(right_pano)

    # if width_right_pano > width_left_pano, Select right_pano as destination. Otherwise is left_pano
    if right_pano.shape[1] >= left_pano.shape[1]:
        H = np.array([[ 9.89237566e-01, -8.31836475e-03,  1.03405200e+03],
                      [-1.15646888e-03,  9.96171969e-01,  6.23854009e-01],
                      [-5.22492193e-06, -5.11404809e-06,  1.00000000e+00]])
        fullpano, _, _, _ = warpTwoImages(left_pano, right_pano, iteration, H)
    else:
        H = np.array([[ 9.89237566e-01, -8.31836475e-03,  1.03405200e+03],
                      [-1.15646888e-03,  9.96171969e-01,  6.23854009e-01],
                      [-5.22492193e-06, -5.11404809e-06,  1.00000000e+00]])
        fullpano, _, _, _ = warpTwoImages(right_pano, left_pano, iteration, H)
    return fullpano


def crop(panorama, h_dst, conners):
    """crop panorama based on destination.
    @param panorama is the panorama
    @param h_dst is the hight of destination image
    @param conner is the tuple which containing 4 conners of warped image and
    4 conners of destination image"""
    # find max min of x,y coordinate
    [xmin, ymin] = np.int32(conners.min(axis=0).ravel() - 0.5)
    t = [-xmin, -ymin]
    conners = conners.astype(int)

    # conners[0][0][0] is the X coordinate of top-left point of warped image
    # If it has value<0, warp image is merged to the left side of destination image
    # otherwise is merged to the right side of destination image
    if conners[0][0][0] < 0:
        n = abs(-conners[1][0][0] + conners[0][0][0])
        panorama = panorama[t[1] : h_dst + t[1], n:, :]
    else:
        if conners[2][0][0] < conners[3][0][0]:
            panorama = panorama[t[1] : h_dst + t[1], 0 : conners[2][0][0], :]
        else:
            panorama = panorama[t[1] : h_dst + t[1], 0 : conners[3][0][0], :]
    return panorama
