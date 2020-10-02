# Scripts to try and detect key frames that represent scene transitions
# in a video. Has only been tried out on video of slides, so is likely not
# robust for other types of video.

import cv2
# import cv
import argparse
import json
import os
import numpy as np
import errno
from blur_detection import BlurDetection
from estimate_brightness import EstimateBrightness
from estimate_darkness import EstimateDarkness


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def getInfo(sourcePath):
    cap = cv2.VideoCapture(sourcePath)
    info = {
        "framecount": cap.get(cv2.CAP_PROP_FRAME_COUNT),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "codec": int(cap.get(cv2.CAP_PROP_FOURCC))
    }
    cap.release()
    return info


def scale(img, xScale, yScale):
    res = cv2.resize(img, None, fx=xScale, fy=yScale, interpolation=cv2.INTER_AREA)
    return res


def resize(img, width, height):
    res = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return res


#
# Extract [numCols] domninant colors from an image
# Uses KMeans on the pixels and then returns the centriods
# of the colors
#
def extract_cols(image, numCols):
    # convert to np.float32 matrix that can be clustered
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)

    # Set parameters for the clustering
    max_iter = 20
    epsilon = 1.0
    K = numCols
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)

    # cluster
    lab = []
    compactness, labels, centers = cv2.kmeans(data=Z, K=K, bestLabels=None, criteria=criteria, attempts=10,
                                              flags=cv2.KMEANS_RANDOM_CENTERS)

    clusterCounts = []
    for idx in range(K):
        mask = labels[:] == idx
        count = np.sum(mask)
        # print(f"Number of pixels at K:{idx} = {count}")
        # the original code raises a numpy error about dimensionality
        # count = len(Z[labels == idx])
        clusterCounts.append(count)

    # Reverse the cols stored in centers because cols are stored in BGR
    # in opencv.
    rgbCenters = []
    for center in centers:
        bgr = center.tolist()
        bgr.reverse()
        rgbCenters.append(bgr)

    cols = []
    for i in range(K):
        iCol = {
            "count": clusterCounts[i],
            "col": rgbCenters[i]
        }
        cols.append(iCol)

    return cols


#
# Calculates change data one one frame to the next one.
#
def calculateFrameStats(sourcePath, verbose=False, after_frame=0):
    cap = cv2.VideoCapture(sourcePath)

    data = {
        "frame_info": []
    }

    # TODO: faster, better ways of removing frames...

    blur_detector = BlurDetection(threshold=14, blur_detection_method="variance_of_laplacian")
    brightness_estimator = EstimateBrightness(white_threshold=220, empty_pixels_allowed=0)
    darkness_estimator = EstimateDarkness(black_threshold=150, dark_pixels_allowed=0)

    lastFrame = None
    while (cap.isOpened()):
        ret, frame = cap.read()
        if frame is None:
            break

        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1

        # Convert to grayscale
        gray = BlurDetection.to_gray(frame)

        # estimate blurriness
        blurriness = blur_detector.measure_of_focus(gray)

        # estimate brightness
        brightness = brightness_estimator.measure_of_brightness(gray)
        darkness = darkness_estimator.measure_of_darkness(gray)

        # estimate average intensity
        cols, rows = gray.shape
        num_pixels = cols * rows
        intensity = np.sum(gray) / (255 * num_pixels)

        # Scale down and blur to make image differences more robust to noise
        gray = scale(gray, 0.25, 0.25)
        gray = cv2.GaussianBlur(gray, (9, 9), 0.0)

        if frame_number < after_frame:
            lastFrame = gray
            continue

        if lastFrame is not None:
            # simple subtraction keeps identical frames, but with less light (i.e. same, but darker)
            # TODO: this belongs refactored into the metadata, then acted upon in DetectScenes...
            lastFrame_intensity = np.sum(lastFrame) / (255 * num_pixels)
            gray_intensity = np.sum(gray) / (255 * num_pixels)
            adjusted_lastFrame = cv2.convertScaleAbs(lastFrame, alpha=(gray_intensity / lastFrame_intensity), beta=0)

            diff = cv2.subtract(gray, adjusted_lastFrame)

            diffMag = cv2.countNonZero(diff)

            frame_info = {
                "frame_number": int(frame_number),
                "diff_count": int(diffMag),
                "blurriness": float(blurriness),
                "brightness": float(brightness),
                "darkness": float(darkness),
                "intensity": float(intensity)
            }
            data["frame_info"].append(frame_info)

            if verbose:
                cv2.imshow('diff', diff)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Keep a ref to his frame for differencing on the next iteration
        lastFrame = gray

    cap.release()
    cv2.destroyAllWindows()

    # compute some stats
    update_stats(data, "diff_count")
    update_stats(data, "blurriness")
    update_stats(data, "brightness")
    update_stats(data, "darkness")

    return data


#
# Take an image and write it out at various sizes.
#
# TODO: Create output directories if they do not exist.
#
seq_num_global = 0


def update_stats(data_to_update, desired_attribute: str):
    if data_to_update is None or desired_attribute is None \
            or "frame_info" not in data_to_update:       # data is a dictionary
        return False
    # be sure we have a place for new stats
    if "stats" not in data_to_update:
        data_to_update["stats"] = {}
    # reshape data for this attribute
    desired_attribute_array = [frame_info[desired_attribute] for frame_info in data_to_update["frame_info"]]
    data_to_update["stats"][desired_attribute] = {
        "num": len(desired_attribute_array),
        "min": np.min(desired_attribute_array),
        "max": np.max(desired_attribute_array),
        "mean": np.mean(desired_attribute_array),
        "median": np.median(desired_attribute_array),
        "sd": np.std(desired_attribute_array)
    }
    greater_than_mean = [fi for fi in data_to_update["frame_info"] if fi[desired_attribute] > data_to_update["stats"][desired_attribute]["mean"]]
    greater_than_median = [fi for fi in data_to_update["frame_info"] if fi[desired_attribute] > data_to_update["stats"][desired_attribute]["median"]]
    greater_than_one_sd = [fi for fi in data_to_update["frame_info"] if
                           fi[desired_attribute] > data_to_update["stats"][desired_attribute]["sd"] + data_to_update["stats"][desired_attribute]["mean"]]
    greater_than_two_sd = [fi for fi in data_to_update["frame_info"] if
                           fi[desired_attribute] > (data_to_update["stats"][desired_attribute]["sd"] * 2) + data_to_update["stats"][desired_attribute]["mean"]]
    greater_than_three_sd = [fi for fi in data_to_update["frame_info"] if
                             fi[desired_attribute] > (data_to_update["stats"][desired_attribute]["sd"] * 3) + data_to_update["stats"][desired_attribute]["mean"]]

    data_to_update["stats"][desired_attribute]["greater_than_mean"] = len(greater_than_mean)
    data_to_update["stats"][desired_attribute]["greater_than_median"] = len(greater_than_median)
    data_to_update["stats"][desired_attribute]["greater_than_one_sd"] = len(greater_than_one_sd)
    data_to_update["stats"][desired_attribute]["greater_than_three_sd"] = len(greater_than_three_sd)
    data_to_update["stats"][desired_attribute]["greater_than_two_sd"] = len(greater_than_two_sd)

    return data_to_update


def writeImagePyramid(destPath, name, seqNumber, image, border_color):
    global seq_num_global
    fullPath = os.path.join(destPath, "full", name + "-" + str(seqNumber).zfill(4) + ".png")
    fullSeqPath = os.path.join(destPath, "fullseq", name + "-" + str(seq_num_global).zfill(4) + ".png")
    seq_num_global += 1
    # halfPath = os.path.join(destPath, "half", name + "-" + str(seqNumber).zfill(4) + ".png")
    # quarterPath = os.path.join(destPath, "quarter", name + "-" + str(seqNumber).zfill(4) + ".png")
    # eighthPath = os.path.join(destPath, "eighth", name + "-" + str(seqNumber).zfill(4) + ".png")
    # sixteenthPath = os.path.join(destPath, "sixteenth", name + "-" + str(seqNumber).zfill(4) + ".png")

    # hImage = scale(image, 0.5, 0.5)
    # qImage = scale(image, 0.25, 0.25)
    # eImage = scale(image, 0.125, 0.125)
    # sImage = scale(image, 0.0625, 0.0625)

    # TRY: 2020-10-01
    bordered_image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=border_color)
    cv2.imwrite(fullPath, bordered_image)
    cv2.imwrite(fullSeqPath, bordered_image)
    # cv2.imwrite(halfPath, hImage)
    # cv2.imwrite(quarterPath, qImage)
    # cv2.imwrite(eighthPath, eImage)
    # cv2.imwrite(sixteenthPath, sImage)


#
# Selects a set of frames as key frames (frames that represent a significant difference in
# the video i.e. potential scene changes). Key frames are selected as those frames where the
# number of pixels that changed from the previous frame are more than 1.85 standard deviations
# times from the mean number of changed pixels across all interframe changes.
#
# TODO: misses FIRST frame!!!
def detectScenes(sourcePath, destPath, data, name, verbose=False):
    destDir = os.path.join(destPath, "images")

    # TODO make sd multiplier externally configurable
    diff_threshold = (data["stats"]["diff_count"]["sd"] * 1.0) + data["stats"]["diff_count"]["mean"]
    blur_threshold = (data["stats"]["blurriness"]["sd"] * 2.0) + data["stats"]["blurriness"]["mean"]
    bright_threshold = (data["stats"]["brightness"]["sd"] * 1.0) + data["stats"]["brightness"]["mean"]
    dark_threshold = (data["stats"]["darkness"]["sd"] * 1.0) + data["stats"]["darkness"]["mean"]

    cap = cv2.VideoCapture(sourcePath)
    for index, fi in enumerate(data["frame_info"]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi["frame_number"])
        ret, frame = cap.read()

    # TODO: refactor so all frames are extracted, but selected frames are outlined...
        if fi["brightness"] > bright_threshold:
            writeImagePyramid(destDir, name, fi["frame_number"], frame, border_color=[255, 255, 255])
            continue
        if fi["darkness"] > dark_threshold:
            writeImagePyramid(destDir, name, fi["frame_number"], frame, border_color=[0, 0, 0, ])
            continue
        if fi["blurriness"] > blur_threshold:
            writeImagePyramid(destDir, name, fi["frame_number"], frame, border_color=[255, 0, 0])
            continue
        if fi["diff_count"] < diff_threshold:
            writeImagePyramid(destDir, name, fi["frame_number"], frame, border_color=[0, 0, 255])
            continue

        # extract dominant color
        # TODO: what is this for??
        small = resize(frame, 100, 100)
        cols = extract_cols(small, 5)
        data["frame_info"][index]["dominant_cols"] = cols

        if frame is not None:
            writeImagePyramid(destDir, name, fi["frame_number"], frame, border_color=[0, 255, 0])

            if verbose:
                cv2.imshow('extract', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()
    return data


def makeOutputDirs(path):
    try:
        # todo this doesn't quite work like mkdirp. it will fail
        # fi any folder along the path exists. fix
        os.makedirs(os.path.join(path, "metadata"))
        os.makedirs(os.path.join(path, "images", "full"))
        os.makedirs(os.path.join(path, "images", "fullseq"))
        # os.makedirs(os.path.join(path, "images", "half"))
        # os.makedirs(os.path.join(path, "images", "quarter"))
        # os.makedirs(os.path.join(path, "images", "eigth"))
        # os.makedirs(os.path.join(path, "images", "sixteenth"))
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


parser = argparse.ArgumentParser()

parser.add_argument('-s', '--source', help='source file', required=True)
parser.add_argument('-d', '--dest', help='dest folder', required=True)
parser.add_argument('-n', '--name', help='image sequence name', required=True)
parser.add_argument('-a', '--after_frame', help='after frame', default=0)
parser.add_argument('-v', '--verbose', action='store_true')
parser.set_defaults(verbose=False)

args = parser.parse_args()

if args.verbose:
    info = getInfo(args.source)
    print("Source Info: ", info)

makeOutputDirs(args.dest)

# Run the extraction
data = calculateFrameStats(args.source, args.verbose, int(args.after_frame))
data = detectScenes(args.source, args.dest, data, args.name, args.verbose)
keyframeInfo = [frame_info for frame_info in data["frame_info"] if "dominant_cols" in frame_info]

# Write out the results
data_fp = os.path.join(args.dest, "metadata", args.name + "-meta.json")
with open(data_fp, 'w') as f:
    data_json_str = json.dumps(data, indent=4, cls=NumpyEncoder)
    f.write(data_json_str)

keyframe_info_fp = os.path.join(args.dest, "metadata", args.name + "-keyframe-meta.json")
with open(keyframe_info_fp, 'w') as f:
    data_json_str = json.dumps(keyframeInfo, indent=4, cls=NumpyEncoder)
    f.write(data_json_str)
