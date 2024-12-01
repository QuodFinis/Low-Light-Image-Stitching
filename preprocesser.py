import cv2
import numpy as np


class Preprocesser:
    def clahe(self, image, clip_limit=2.0, tile_grid_size=(8, 8), colorspace="LAB"):
        """
        Improve local contrast while limiting noise amplification in darker regions.

        Parameters:
        - clip_limit (float): Threshold for contrast limiting.
            - Higher values allow more contrast amplification.
            - Example: `clip_limit=2.0` (moderate), `clip_limit=5.0` (strong).
        - tile_grid_size (tuple): Size of the grid for contextual contrast adjustment.
            - Smaller grids (e.g., (4, 4)) enhance local details more but may introduce artifacts.
            - Larger grids (e.g., (16, 16)) smooth results but may lose finer details.
        - colorspace (str): Colorspace for CLAHE.
            - "LAB": Default, adjusts brightness while preserving color.
            - "YCrCb": Alternative for noise suppression in darker areas.
        """
        if colorspace == "LAB":
            converted = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            channels = cv2.split(converted)
        elif colorspace == "YCrCb":
            converted = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            channels = cv2.split(converted)
        else:
            raise ValueError(f"Unsupported colorspace: {colorspace}. Use 'LAB' or 'YCrCb'.")
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        channels[0] = clahe.apply(channels[0])
        processed = cv2.merge(channels)
        return cv2.cvtColor(processed, cv2.COLOR_LAB2BGR if colorspace == "LAB" else cv2.COLOR_YCrCb2BGR)

    def gamma_correction(self, image, gamma=1.0):
        """
        Enhance visibility in dark regions non-linearly.

        Parameters:
        - gamma (float): Non-linear intensity transformation parameter.
            - gamma > 1: Brightens the image (e.g., 2.0 makes the image twice as bright).
            - gamma < 1: Darkens the image (e.g., 0.5 reduces brightness).
        """
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def log_transform(self, image, constant=255):
        """
        Compress the dynamic range of pixels for better visualization of low-light details.

        Parameters:
        - constant (float): Scaling constant for logarithmic transformation.
            - Higher values increase brightness, emphasizing lower intensities.
        """
        c = constant / np.log(1 + np.max(image))
        log_transformed = c * np.log(1 + image.astype(np.float64))
        return np.array(log_transformed, dtype=np.uint8)

    def bilateral_filter(self, image, d=9, sigma_color=75, sigma_space=75):
        """
        Preserve edges while reducing noise.

        Parameters:
        - d (int): Diameter of pixel neighborhood.
            - Larger values process a larger area around each pixel.
        - sigma_color (float): Color difference tolerance.
            - Higher values allow more color smoothing.
        - sigma_space (float): Spatial distance tolerance.
            - Higher values smooth farther pixels.
        """
        return cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    def adaptive_smoothing(self, image, h=10, h_color=10, template_window_size=7, search_window_size=21):
        """
        Smooth darker regions more than brighter regions while retaining high-contrast features.

        Parameters:
        - h (float): Strength of luminance smoothing.
        - h_color (float): Strength of color smoothing.
        - template_window_size (int): Size of template patch.
        - search_window_size (int): Area used to find similar patches.
        """
        return cv2.fastNlMeansDenoisingColored(image, None, h, h_color, template_window_size, search_window_size)

    def multiscale_retinex(self, image, sigma_list=[15, 80, 250]):
        """
        Enhance contrast and brightness while removing shadows and highlights.

        Parameters:
        - sigma_list (list of floats): Scales for Gaussian blurring.
            - Smaller values focus on local details.
            - Larger values focus on global illumination.
        """
        retinex = np.zeros_like(image, dtype=np.float64)
        for sigma in sigma_list:
            blurred = cv2.GaussianBlur(image, (0, 0), sigma)
            retinex += np.log10(image + 1) - np.log10(blurred + 1)
        retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
        return np.uint8(retinex)

    def haze_removal(self, image, w=0.95, t0=0.1, k=0.75):
        """
        Enhance visibility in hazy conditions.

        Parameters:
        - w (float): Weight of dark channel in atmospheric light estimation.
            - Higher values reduce haze more aggressively.
        - t0 (float): Minimum transmission threshold.
            - Prevents complete blackout in very dark regions.
        - k (float): Scaling factor for atmospheric light adjustment.
        """
        min_channel = np.min(image, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dark_channel = cv2.erode(min_channel, kernel)
        dark_channel = cv2.merge([dark_channel] * 3).astype(np.float64)
        haze = 1 - w * dark_channel
        t = 1 - k * dark_channel
        t[t < t0] = t0
        return haze / t[:, :, np.newaxis] + (1 - w)

    def sharpen(self, image, kernel=np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])):
        """
        Enhance edges for sharper image clarity.

        Parameters:
        - kernel (ndarray): Convolution kernel for sharpening.
            - Example: `[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]` enhances edges.
        """
        return cv2.filter2D(image, -1, kernel)

    def color_channel_scaling(self, image, red_scale=1.0, blue_scale=1.0, green_scale=1.0):
        """
        Adjust intensity of individual color channels.

        Parameters:
        - red_scale (float): Scaling factor for red channel.
        - blue_scale (float): Scaling factor for blue channel.
        - green_scale (float): Scaling factor for green channel.
            - Example: `red_scale=1.5, blue_scale=1.2` brightens red and blue channels.
        """
        image = image.astype(np.float64)
        image[:, :, 2] *= red_scale
        image[:, :, 0] *= blue_scale
        image[:, :, 1] *= green_scale
        return np.clip(image, 0, 255).astype("uint8")
