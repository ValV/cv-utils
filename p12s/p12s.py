# p12s - preprocessings for SAR-polarized pairs

import os
import sys
import cv2 as cv
import numpy as np

from glob import glob


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    assert type(image) is np.ndarray and image.ndim in [2, 3], \
    "Image is None or not a 2D or 3D array. Can't resize!"

    height = round(image.shape[0] * width / image.shape[1])

    return cv.resize(image, (width, height))


def draw_text(image: np.ndarray, text: str, x1: int, x2: int, y: int) -> None:
    if not text:
        return None
    assert type(image) is np.ndarray, "Can't draw text on None image!"

    font = cv.FONT_HERSHEY_PLAIN
    scale = 1
    thickness = 1
    margin = 8
    line = cv.LINE_AA
    fg, bg = ((255, 255, 255), (0, 0, 0))

    size, baseline = cv.getTextSize(text, font, scale, thickness)
    if x2:
        x = x2 - (x2 - x1) // 2 - size[0] // 2
    else:
        x = x1 + margin
    y = y + size[1] + margin

    cv.putText(image, text, (x - 1, y - 1), font, scale, bg, thickness, line)
    cv.putText(image, text, (x + 1, y - 1), font, scale, bg, thickness, line)
    cv.putText(image, text, (x + 1, y + 1), font, scale, bg, thickness, line)
    cv.putText(image, text, (x - 1, y + 1), font, scale, bg, thickness, line)

    cv.putText(image, text, (x, y), font, scale, fg, thickness, line)

    return None


def pin_light(blend: np.ndarray, target: np.ndarray) -> np.ndarray:
    # Proposed usage pin_light(HV, HH), 'cause HH is usually brighter
    assert type(blend) is np.ndarray and type(target) is np.ndarray and \
           blend.shape == target.shape, \
    "Blend or target is None, or their shapes mismatch!"

    return (blend > 127) * np.maximum(target, 2 * (blend - 127)) + \
           (blend <= 127) * np.minimum(target, 2 * blend)


def log_levels(image: np.ndarray) -> np.ndarray:
    assert type(image) is np.ndarray and \
           ((image.ndim == 3 and image.shape[-1] == 3) or image.ndim == 2), \
    "Image is None or has a wring channels number!"

    lut = (np.geomspace(256, 1, 256, endpoint=True, dtype=np.int16) * -1 + 256)\
          .astype(np.uint8)

    if image.ndim == 3:
        image[..., 0] = cv.LUT(image[..., 0], lut)
        image[..., 1] = cv.LUT(image[..., 1], lut)
        image[..., 2] = cv.LUT(image[..., 2], lut)
    else:
        image = cv.LUT(image, lut)

    return image


def rgb_mux(p1_image: np.ndarray, p2_image: np.ndarray) -> np.ndarray:
    assert type(p1_image) is np.ndarray and type(p2_image) is np.ndarray and \
           p1_image.ndim == p2_image.ndim == 2, \
    "Can't multiplex None images or non-2D images!"
    return np.stack((p1_image, pin_light(p2_image, p1_image), p2_image), axis=2)


def process_pair(p1_image: np.ndarray, p2_image: np.ndarray) -> np.ndarray:
    assert type(p1_image) is np.ndarray and type(p2_image) is np.ndarray and \
           p1_image.shape == p2_image.shape, \
    "Polarization images are None or their shapes mismatch!"

    p1, p2 = (resize_to_width(p1_image, 400), resize_to_width(p2_image, 400))
    height, width = p1.shape[:2]

    def idx(row: int, col: int) -> (int, int):
        # Calculate a slice for the current row and column in pixels
        return (slice(height * (row - 1), height * row),
                slice(width * (col - 1), width * col))

    # Output image
    rows, cols = (9, 2) # cols are supposed to be always = 2
    image = np.zeros((height * rows, width * cols, 3), dtype=np.uint8)

    # Intermediate images (the same size as original)
    image_p1_equal = cv.equalizeHist(p1_image)
    image_p2_equal = cv.equalizeHist(p2_image)

    image_p1_log = log_levels(p1_image)
    image_p2_log = log_levels(p2_image)

    image_pinned = pin_light(p2_image, p1_image)
    image_logpeq = cv.equalizeHist(pin_light(image_p2_log, image_p1_log))

    # 1st row (original)
    roi = idx(1, 1)
    image[roi] = np.repeat(p1[..., None], 3, axis=2)
    draw_text(image, "Original HH", roi[1].start, roi[1].stop, roi[0].start)

    roi = idx(1, 2)
    image[roi] = np.repeat(p2[..., None], 3, axis=2)
    draw_text(image, "Original HV", roi[1].start, roi[1].stop, roi[0].start)

    # 2nd row (equalize)
    roi = idx(2, 1)
    image[roi] = np.repeat(
        resize_to_width(image_p1_equal, 400)[..., None], 3, axis=2)
    draw_text(image, "Equalize HH",
              roi[1].start, roi[1].stop, roi[0].start)

    roi = idx(2, 2)
    image[roi] = np.repeat(
        resize_to_width(image_p2_equal, 400)[..., None], 3, axis=2)
    draw_text(image, "Equalize HV",
              roi[1].start, roi[1].stop, roi[0].start)

    # 3rd row (logarithmize)
    roi = idx(3, 1)
    image[roi] = np.repeat(
        resize_to_width(image_p1_log, 400)[..., None], 3, axis=2)
    draw_text(image, "Logspace HH",
              roi[1].start, roi[1].stop, roi[0].start)

    roi = idx(3, 2)
    image[roi] = np.repeat(
        resize_to_width(image_p2_log, 400)[..., None], 3, axis=2)
    draw_text(image, "Logspace HV",
              roi[1].start, roi[1].stop, roi[0].start)

    # 4th row (logarithmize + equalize)
    roi = idx(4, 1)
    image[roi] = np.repeat(
        resize_to_width(cv.equalizeHist(image_p1_log), 400)[..., None],
        3, axis=2)
    draw_text(image, "Logspace + Equalize HH",
              roi[1].start, roi[1].stop, roi[0].start)

    roi = idx(4, 2)
    image[roi] = np.repeat(
        resize_to_width(cv.equalizeHist(image_p2_log), 400)[..., None],
        3, axis=2)
    draw_text(image, "Logspace + Equalize HV",
              roi[1].start, roi[1].stop, roi[0].start)

    # 5th row (pin_light, logarithmize + pin_light + equalize)
    roi = idx(5, 1)
    image[roi] = np.repeat(
        resize_to_width(image_pinned, 400)[..., None], 3, axis=2)
    draw_text(image, "PinLight(HV, HH)",
              roi[1].start, roi[1].stop, roi[0].start)

    roi = idx(5, 2)
    image[roi] = np.repeat(
        resize_to_width(image_logpeq, 400)[..., None], 3, axis=2)
    draw_text(image, "Logspace + PinLight(HV, HH) + Equalize",
              roi[1].start, roi[1].stop, roi[0].start)

    # 6th row (logarithmize + pin_light, pin_light + logarithmize)
    roi = idx(6, 1)
    image[roi] = np.repeat(
        resize_to_width(pin_light(image_p2_log, image_p1_log), 400)[..., None],
        3, axis=2)
    draw_text(image, "Logspace + PinLight(HV, HH)",
              roi[1].start, roi[1].stop, roi[0].start)

    roi = idx(6, 2)
    image[roi] = np.repeat(
        resize_to_width(log_levels(image_pinned), 400)[..., None], 3, axis=2)
    draw_text(image, "PinLight(HV, HH) + Logspace",
              roi[1].start, roi[1].stop, roi[0].start)

    # 7th row (equalize + pin_light, pin_light + equalize)
    roi = idx(7, 1)
    image[roi] = np.repeat(
        resize_to_width(
            pin_light(image_p2_equal, image_p1_equal), 400)[..., None],
        3, axis=2)
    draw_text(image, "Equaize + PinLight(HV, HH)",
              roi[1].start, roi[1].stop, roi[0].start)

    roi = idx(7, 2)
    image[roi] = np.repeat(
        resize_to_width(cv.equalizeHist(image_pinned), 400)[..., None],
        3, axis=2)
    draw_text(image, "PinLight(HV, HH) + Equalize",
              roi[1].start, roi[1].stop, roi[0].start)

    # 8th row (equalize + pin_light, pin_light + equalize)
    roi = idx(8, 1)
    image[roi] = resize_to_width(rgb_mux(image_p1_log, image_p2_log), 400)
    draw_text(image, "Logspace + RGB mux",
              roi[1].start, roi[1].stop, roi[0].start)

    roi = idx(8, 2)
    image[roi] = resize_to_width(rgb_mux(image_p1_equal, image_p2_equal), 400)
    draw_text(image, "Equalize + RGB mux",
              roi[1].start, roi[1].stop, roi[0].start)

    # 9th row (equalize + pin_light, pin_light + equalize)
    roi = idx(9, 1)
    image[roi] = resize_to_width(rgb_mux(cv.equalizeHist(image_p1_log),
                                         cv.equalizeHist(image_p2_log)), 400)
    draw_text(image, "Logspace + Equalize + RGB mux",
              roi[1].start, roi[1].stop, roi[0].start)

    roi = idx(9, 2)
    image[roi] = resize_to_width(rgb_mux(p1_image, p2_image), 400)
    draw_text(image, "RGB mux",
              roi[1].start, roi[1].stop, roi[0].start)

    return image


def process_all(source: str, destination: str = None, preview: bool = True,
                p1: str = "hh", p2: str = "hv") -> None:
    if destination:
        os.makedirs(destination, exist_ok=True)
        assert os.access(destination, os.W_OK), "Can't write to destination!"

    print(f"DEBUG: source = {source}, destination = {destination}")
    p1_files = sorted(glob(os.path.join(source, f"*{p1}*")))
    p2_files = sorted(glob(os.path.join(source, f"*{p2}*")))
    # print(f"DEBUG: input images (HH --- HV) size = {len(p1_files)}")

    for pair_files in zip(p1_files, p2_files):
        # print(*pair_files, sep=" --- ")
        p1_file = os.path.basename(pair_files[0])
        p2_file = os.path.basename(pair_files[1])
        # print(p1_file, p2_file, sep=" --- ")

        p1_image = cv.imread(pair_files[0], cv.IMREAD_GRAYSCALE)
        p2_image = cv.imread(pair_files[1], cv.IMREAD_GRAYSCALE)
        # print(p1_image.shape, p2_image.shape, sep=" --- ")

        image = process_pair(p1_image, p2_image)

        if destination:
            out_file = p1_file.replace(p1, "p12s")
            cv.imwrite(os.path.join(destination, out_file), image)

        if preview:
            print("Showing preprocessed images. Press 'ESC' to exit",
                  "or 'Space' to preview next")
            cv.imshow(f"{p1_file} <---> {p2_file}", image)

            while True:
                key = cv.waitKey()
                if key == 27:
                    # Return if Esc is pressed
                    cv.destroyAllWindows()
                    return None
                elif key == 32:
                    # Go to next pair if Space is pressed
                    break
            cv.destroyAllWindows()

    return None


if __name__ == "__main__":
    assert 1 < len(sys.argv) < 4 and os.path.exists(sys.argv[1]), \
    "Provide a valid path for SAR-polarized image pairs!"

    process_all(*sys.argv[1:], preview=False)
