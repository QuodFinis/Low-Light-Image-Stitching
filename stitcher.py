import cv2
import numpy as np


class Stitcher:
    def stitch(self, images: list[str], ratio=0.75, reprojThresh=4.0, showMatches=False):
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
        # unpack the images, then detect keypoints and extract local invariant descriptors from them
        (imageB, imageA) = images
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # match features between the two images, return the matches and homography and status which indicates which keypoints in matches were successfully spatially verified using RANSAC
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

        # if the match is None, then there aren't enough matched keypoints to create a panorama
        if M is None:
            return None

        # otherwise, apply a perspective warp to stitch the images together
        (matches, H, status) = M

        # warp right image to shape of output image (sum of widths of both images and height of left image)
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        # Ensure that result has at least the same size as imageB in the first two dimensions
        min_height = min(result.shape[0], imageB.shape[0])
        min_width = min(result.shape[1], imageB.shape[1])
        # Crop result and imageB to the minimum dimensions
        result[:min_height, :min_width] = imageB[:min_height, :min_width]

        # check to see if the keypoint matches should be visualized
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            # return a tuple of the stitched image and the visualization
            return (result, vis)

        # return the stitched image
        return result

    def detectAndDescribe(self, image):
        """
        Detects keypoints and extracts local invariant descriptors from an image using Difference of Gaussian (DoG) keypoint detector and SIFT feature extractor.

        :param image:
            Image to detect keypoints and extract features from
        :return:
            Tuple of keypoints and features
        """

        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect and extract features from the image
        descriptor = cv2.SIFT_create()
        (kps, features) = descriptor.detectAndCompute(image, None)

        # convert the keypoints from KeyPoint objects to NumPy arrays
        kps = np.float32([kp.pt for kp in kps])

        # return a tuple of keypoints and features
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        """
        Matches features by comparing their local invariant descriptors. Matches features by looping over descriptors of both images, computing their distance and finding the smallest Euclidean distance between all feature vectors for each pair. Uses k-NN matching between the feature vectors to find the best two matches for each feature vector. Applies David Lowe's ratio test to filter out false positives, keeping only high-quality feautre matches. Compute the homography between the two sets of keypoints (requireing at least 4 matches). Uses RANSAC to estimate the homography matrix and remove outliers.

        :param kpsA:
            Keypoints from the first image
        :param kpsB:
            Keypoints from the second image
        :param featuresA:
            Features from the first image as vector
        :param featuresB:
            Features from the second image as v ector
        :param ratio:
            Ratio for David Lowe's ratio test when matching features
        :param reprojThresh:
            Maximum pixel wiggle room allowed by RANSAC
        :return:
            Tuple of matches, homography, and status
        """

        # compute the raw matches and initialize the list of actual matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

            # return the matches, homography, and status
            return (matches, H, status)

        # otherwise, no homography could be computed
        return None


    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        """
        Draws keypoint matches on the images by drawing lines between the matched keypoints.

        :param imageA:
            Left image
        :param imageB:
            Right image
        :param kpsA:
            Keypoints from the left image as vector
        :param kpsB:
            Keypoints from the right image as vector
        :param matches:
            Matches between the keypoints after applying Lowe's ratio test
        :param status:
            Status of the matches after applying RANSAC for homography estimation
        :return:
            Visualization of the matches
        """

        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # return the visualization
        return vis

