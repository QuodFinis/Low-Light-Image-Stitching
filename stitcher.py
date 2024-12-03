import cv2
import numpy as np
from imutils import paths
from preprocesser import Preprocesser


class Stitcher:
    def __init__(self, preprocessor: Preprocesser = None, ratio=0.75, reprojThresh=4.0, blending_factor=0.5, showMatches=False):
        """
        Initializes the stitcher with an optional preprocessor to apply to the images before stitching.

        :param preprocessor:
            Preprocessor to apply to the images before stitching
        """
        # store the image preprocessor
        self.preprocessor = preprocessor

    def stitch(self, images_dir, output_path):
        """
        Stitches two images together by detecting keypoints and extracting local invariant descriptors from them. Matches features between the two images, applies a perspective warp to stitch the images together, and returns the stitched image.

        :param images:
            List of two images to stitch together in left to right order (reverse order will result in output ___ image)
        :param ratio:
            Ratio for David Lowe's ratio test when matching features
        :param reprojThresh:
            Maximum pixel wiggle room allowed by RANSAC
        :param showMatches:
            Whether to display keypoint matches
        :return:
            Stitched image
        """

        print("[INFO] loading images...")
        imagePaths = sorted(list(paths.list_images(images_dir)))
        images = []

        # loop over the image paths, load each one, and add them to our images to stitch list
        for imagePath in imagePaths:
            image = cv2.imread(imagePath)
            images.append(image)

        # initialize OpenCV's image stitcher object and then perform the image stitching
        print("[INFO] stitching images...")
        stitcher = cv2.Stitcher.create()
        (status, stitched) = stitcher.stitch(images)

        # if the status is '0', then OpenCV successfully performed image stitching
        if status == 0:
            # write the output stitched image to disk
            cv2.imwrite(output_path, stitched)

            # display the output stitched image to our screen
            cv2.imshow("Stitched", stitched)
            cv2.waitKey(0)

        # otherwise the stitching failed, likely due to not enough keypoints being detected
        else:
            print("[INFO] image stitching failed ({})".format(status))

        # close all windows
        cv2.destroyAllWindows()


    def metrics(self, matches, status, imageA, kpsA, imageB, stitched_image):
        # Metrics computation
        num_matches = len(matches)
        num_inliers = np.sum(status)
        matching_ratio = num_inliers / num_matches if num_matches > 0 else 0

        # Compactness: Percentage of the image covered by keypoints
        keypoint_areas = np.zeros(imageA.shape[:2], dtype=np.uint8)
        for kp in kpsA:
            x, y = int(kp[0]), int(kp[1])
            cv2.rectangle(keypoint_areas, (x - 1, y - 1), (x + 1, y + 1), 1, -1)
        compactness = np.sum(keypoint_areas) / (imageA.shape[0] * imageA.shape[1])

        # Edge Similarity Index (ESI)
        edgesA = cv2.Canny(cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY), 100, 200)
        edgesB = cv2.Canny(cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY), 100, 200)
        edges_stitched = cv2.Canny(cv2.cvtColor(stitched_image, cv2.COLOR_BGR2GRAY), 100, 200)
        esi = cv2.matchTemplate(edges_stitched, edgesA, cv2.TM_CCOEFF_NORMED).max()

        # Structural Similarity Index (SSIM)
        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        grayStitched = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2GRAY)
        (ssim, _) = cv2.quality.QualitySSIM_compute(grayA, grayStitched)

        # PSNR
        psnr = cv2.PSNR(imageA, stitched_image)

        # Compile metrics
        metrics = {
            "num_matches": num_matches,
            "num_inliers": num_inliers,
            "matching_ratio": matching_ratio,
            "compactness": compactness,
            "esi": esi,
            "ssim": ssim[0] if len(ssim) > 0 else 0,
            "psnr": psnr,
        }