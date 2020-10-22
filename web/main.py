from __future__ import division
from __future__ import print_function
from flask import Flask, render_template, jsonify, request
import uuid
import os
import re
import pickle
import sys
import time
import datetime
import traceback
import cv2
import numpy as np
import scipy.optimize

import pandas as pd
import pytesseract
import tensorflow as tf
from pytesseract import Output
from tqdm import tqdm
from fuzzywuzzy import fuzz

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from builtins import zip
from builtins import str
from builtins import range
from builtins import object
from past.utils import old_div

from PIL import Image, ImageEnhance

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

UPLOAD_FOLDER = 'static/img/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# CTPN SETUP
from nets import model_train as model
from utils.rpn_msr.proposal_layer import proposal_layer
from utils.text_connector.detectors import TextDetector
textdetector = TextDetector(DETECT_MODE='O')
output_path = UPLOAD_FOLDER
gpu = '0'
checkpoint_path = 'checkpoints_mlt/'

app = Flask(__name__)

with open('db/ingredient_idx.pickle', 'rb') as handle:
    ingredient_idx = pickle.load(handle)

with open('db/ingredient_idx_1000.pickle', 'rb') as handle:
    ingredient_idx_1000 = pickle.load(handle)

with open('db/rf_200.pkl', 'rb') as mdrf:
    reload_rf = pickle.load(mdrf)

with open('db/recommendation.pkl', 'rb') as recomm:
    reload_knn = pickle.load(recomm)

df_recommendation = pd.read_csv('db/recommendation_pool.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/', methods=['POST'])
def predict():
    file = request.files['file']

    reqform = request.form
    dwstr = reqform.get("dewarp")
    lgstr = reqform.get("lang")
    lang_set = "Vietnamese" if "Vietnamese" in lgstr else "English"

    is_dewarp = True if "true" in dwstr else False

    fname = file.filename
    if fname == "" or not allowed_file(fname):
        return jsonify({})
    img_raw = file.read()
    if not img_raw:
        return jsonify({})

    img, fullpath = process_img(img_raw, ext(fname))
    dw_img_path = ""
    if is_dewarp:
        dw_img_path = page_dewarp(fullpath, output_path=UPLOAD_FOLDER)
    else:
        dw_img_path = preprocess_for_ocr(fullpath, UPLOAD_FOLDER)

    ctpn_img, ctpn_txt = ctpn(dw_img_path)
    res, model_out = ocr_everything(dw_img_path, ctpn_txt, 'db/ingredient_inci_1570.csv', 'db/ingredient_vietnamese_3818.csv', 'db/ingredient_cosing_37309.csv', lang_set, debug=True)

    input_rf, input_knn = transform(model_out)

    prediction = reload_rf.predict(input_rf)

    distances, indices = reload_knn.kneighbors(input_knn)
    location = indices.tolist()[0]
    df_recom = df_recommendation.iloc[location, :]

    pred_list = prediction.tolist()
    score = str(pred_list[0]) if len(pred_list) > 0 else "N/A"

    # Get rid of NaNs
    res = res.fillna("")
    df_recom = df_recom.fillna("")

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    return jsonify({'filename': ctpn_img, 'score': score, 'recom': df_recom.to_dict(orient='records'), 'res': res.to_dict(orient='records')})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def ext(filename):
    return filename.rsplit('.', 1)[1].lower()

def process_img(img_data, ext):
    if not img_data:
        return
    filename = "{}.{}".format(uuid.uuid4().hex, ext)
    fullpath = os.path.join(UPLOAD_FOLDER, filename)
    with open(fullpath, "wb") as f:
        f.write(img_data)
    return img_data, fullpath

# PREPROCESS FOR OCR
# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

# thresholding
def thresholding(image):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 4)

def preprocess_for_ocr(imgfile, output_folder, enhance=1):
    """
    @param img: image to which the pre-processing steps being applied
    """
    img = cv2.imread(imgfile)
    if enhance > 1:
        img = Image.fromarray(img)
        contrast = ImageEnhance.Contrast(img)
        img = contrast.enhance(enhance)
        img = np.asarray(img)

    gray = get_grayscale(img)
    blur = remove_noise(gray)
    res = thresholding(blur)

    img = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)

    namesplit = os.path.splitext(os.path.basename(imgfile))
    basename = namesplit[0] + "_preproc" + namesplit[1]
    out_fullpath = os.path.join(output_folder, basename)

    cv2.imwrite(out_fullpath, img)

    return out_fullpath

# PAGE DEWARP CODE
# pylint: disable=E1101

PAGE_MARGIN_X = 35       # reduced px to ignore near L/R edge
PAGE_MARGIN_Y = 20       # reduced px to ignore near T/B edge

OUTPUT_ZOOM = 1.0        # how much to zoom output relative to *original* image
OUTPUT_DPI = 300         # just affects stated DPI of PNG, not appearance
REMAP_DECIMATE = 16      # downscaling factor for remapping image

ADAPTIVE_WINSZ = 55      # window size for adaptive threshold in reduced px

TEXT_MIN_WIDTH = 15      # min reduced px width of detected text contour
TEXT_MIN_HEIGHT = 2      # min reduced px height of detected text contour
TEXT_MIN_ASPECT = 1.5    # filter out text contours below this w/h ratio
TEXT_MAX_THICKNESS = 10  # max reduced px thickness of detected text contour

EDGE_MAX_OVERLAP = 1.0   # max reduced px horiz. overlap of contours in span
EDGE_MAX_LENGTH = 100.0  # max reduced px length of edge connecting contours
EDGE_ANGLE_COST = 10.0   # cost of angles in edges (tradeoff vs. length)
EDGE_MAX_ANGLE = 7.5     # maximum change in angle allowed between contours

RVEC_IDX = slice(0, 3)   # index of rvec in params vector
TVEC_IDX = slice(3, 6)   # index of tvec in params vector
CUBIC_IDX = slice(6, 8)  # index of cubic slopes in params vector

SPAN_MIN_WIDTH = 30      # minimum reduced px width for span
SPAN_PX_PER_STEP = 20    # reduced px spacing for sampling along spans
FOCAL_LENGTH = 1.2       # normalized focal length of camera

DEBUG_LEVEL = 0          # 0=none, 1=some, 2=lots, 3=all
DEBUG_OUTPUT = 'file'    # file, screen, both

WINDOW_NAME = 'Dewarp'   # Window name for visualization

# nice color palette for visualizing contours, etc.
CCOLORS = [
    (255, 0, 0),
    (255, 63, 0),
    (255, 127, 0),
    (255, 191, 0),
    (255, 255, 0),
    (191, 255, 0),
    (127, 255, 0),
    (63, 255, 0),
    (0, 255, 0),
    (0, 255, 63),
    (0, 255, 127),
    (0, 255, 191),
    (0, 255, 255),
    (0, 191, 255),
    (0, 127, 255),
    (0, 63, 255),
    (0, 0, 255),
    (63, 0, 255),
    (127, 0, 255),
    (191, 0, 255),
    (255, 0, 255),
    (255, 0, 191),
    (255, 0, 127),
    (255, 0, 63),
]

# default intrinsic parameter matrix
K = np.array([
    [FOCAL_LENGTH, 0, 0],
    [0, FOCAL_LENGTH, 0],
    [0, 0, 1]], dtype=np.float32)


def debug_show(name, step, text, display):

    if DEBUG_OUTPUT != 'screen':
        filetext = text.replace(' ', '_')
        outfile = name + '_debug_' + str(step) + '_' + filetext + '.png'
        cv2.imwrite(outfile, display)

    if DEBUG_OUTPUT != 'file':

        image = display.copy()
        height = image.shape[0]

        cv2.putText(image, text, (16, height - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 0), 3, cv2.LINE_AA)

        cv2.putText(image, text, (16, height - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, image)

        while cv2.waitKey(5) < 0:
            pass


def round_nearest_multiple(i, factor):
    i = int(i)
    rem = i % factor
    if not rem:
        return i
    else:
        return i + factor - rem


def pix2norm(shape, pts):
    height, width = shape[:2]
    scl = 2.0 / (max(height, width))
    offset = np.array([width, height], dtype=pts.dtype).reshape((-1, 1, 2)) * 0.5
    return (pts - offset) * scl


def norm2pix(shape, pts, as_integer):
    height, width = shape[:2]
    scl = max(height, width) * 0.5
    offset = np.array([0.5 * width, 0.5 * height],
                      dtype=pts.dtype).reshape((-1, 1, 2))
    rval = pts * scl + offset
    if as_integer:
        return (rval + 0.5).astype(int)
    else:
        return rval


def fltp(point):
    return tuple(point.astype(int).flatten())


def draw_correspondences(img, dstpoints, projpts):

    display = img.copy()
    dstpoints = norm2pix(img.shape, dstpoints, True)
    projpts = norm2pix(img.shape, projpts, True)

    for pts, color in [(projpts, (255, 0, 0)),
                       (dstpoints, (0, 0, 255))]:

        for point in pts:
            cv2.circle(display, fltp(point), 3, color, -1, cv2.LINE_AA)

    for point_a, point_b in zip(projpts, dstpoints):
        cv2.line(display, fltp(point_a), fltp(point_b),
                 (255, 255, 255), 1, cv2.LINE_AA)

    return display


def get_default_params(corners, ycoords, xcoords):

    # page width and height
    page_width = np.linalg.norm(corners[1] - corners[0])
    page_height = np.linalg.norm(corners[-1] - corners[0])
    rough_dims = (page_width, page_height)

    # our initial guess for the cubic has no slope
    cubic_slopes = [0.0, 0.0]

    # object points of flat page in 3D coordinates
    corners_object3d = np.array([
        [0, 0, 0],
        [page_width, 0, 0],
        [page_width, page_height, 0],
        [0, page_height, 0]])

    # estimate rotation and translation from four 2D-to-3D point
    # correspondences
    _, rvec, tvec = cv2.solvePnP(corners_object3d,
                                 corners, K, np.zeros(5))

    span_counts = [len(xc) for xc in xcoords]

    params = np.hstack((np.array(rvec).flatten(),
                        np.array(tvec).flatten(),
                        np.array(cubic_slopes).flatten(),
                        ycoords.flatten()) + tuple(xcoords))

    return rough_dims, span_counts, params


def project_xy(xy_coords, pvec):

    # get cubic polynomial coefficients given
    #
    #  f(0) = 0, f'(0) = alpha
    #  f(1) = 0, f'(1) = beta

    alpha, beta = tuple(pvec[CUBIC_IDX])

    poly = np.array([
        alpha + beta,
        -2 * alpha - beta,
        alpha,
        0])

    xy_coords = xy_coords.reshape((-1, 2))
    z_coords = np.polyval(poly, xy_coords[:, 0])

    objpoints = np.hstack((xy_coords, z_coords.reshape((-1, 1))))

    image_points, _ = cv2.projectPoints(objpoints,
                                        pvec[RVEC_IDX],
                                        pvec[TVEC_IDX],
                                        K, np.zeros(5))

    return image_points


def project_keypoints(pvec, keypoint_index):

    xy_coords = pvec[keypoint_index]
    xy_coords[0, :] = 0

    return project_xy(xy_coords, pvec)


def resize_to_screen(src, maxw=1280, maxh=700, copy=False):

    height, width = src.shape[:2]

    scl_x = float(width) / maxw
    scl_y = float(height) / maxh

    scl = int(np.ceil(max(scl_x, scl_y)))

    if scl > 1.0:
        inv_scl = 1.0 / scl
        img = cv2.resize(src, (0, 0), None, inv_scl, inv_scl, cv2.INTER_AREA)
    elif copy:
        img = src.copy()
    else:
        img = src

    return img


def box(width, height):
    return np.ones((height, width), dtype=np.uint8)


def get_page_extents(small):

    height, width = small.shape[:2]

    xmin = PAGE_MARGIN_X
    ymin = PAGE_MARGIN_Y
    xmax = width - PAGE_MARGIN_X
    ymax = height - PAGE_MARGIN_Y

    page = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(page, (xmin, ymin), (xmax, ymax), (255, 255, 255), -1)

    outline = np.array([
        [xmin, ymin],
        [xmin, ymax],
        [xmax, ymax],
        [xmax, ymin]])

    return page, outline


def get_mask(name, small, pagemask, masktype):

    sgray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

    if masktype == 'text':
        # mask = cv2.adaptiveThreshold(sgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 4)
        mask = cv2.adaptiveThreshold(sgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_WINSZ, 25)

        if DEBUG_LEVEL >= 3:
            debug_show(name, 0.1, 'thresholded', mask)

        mask = cv2.dilate(mask, box(9, 1))

        if DEBUG_LEVEL >= 3:
            debug_show(name, 0.2, 'dilated', mask)

        mask = cv2.erode(mask, box(1, 3))

        if DEBUG_LEVEL >= 3:
            debug_show(name, 0.3, 'eroded', mask)

    else:

        mask = cv2.adaptiveThreshold(sgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV,
                                     ADAPTIVE_WINSZ,
                                     7)

        if DEBUG_LEVEL >= 3:
            debug_show(name, 0.4, 'thresholded', mask)

        mask = cv2.erode(mask, box(3, 1), iterations=3)

        if DEBUG_LEVEL >= 3:
            debug_show(name, 0.5, 'eroded', mask)

        mask = cv2.dilate(mask, box(8, 2))

        if DEBUG_LEVEL >= 3:
            debug_show(name, 0.6, 'dilated', mask)

    return np.minimum(mask, pagemask)


def interval_measure_overlap(int_a, int_b):
    return min(int_a[1], int_b[1]) - max(int_a[0], int_b[0])


def angle_dist(angle_b, angle_a):

    diff = angle_b - angle_a

    while diff > np.pi:
        diff -= 2 * np.pi

    while diff < -np.pi:
        diff += 2 * np.pi

    return np.abs(diff)


def blob_mean_and_tangent(contour):

    moments = cv2.moments(contour)

    area = moments['m00']

    if area == 0:
        return np.array([0, 0]), np.array([0, 0])

    mean_x = old_div(moments['m10'], area)
    mean_y = old_div(moments['m01'], area)

    moments_matrix = old_div(np.array([
        [moments['mu20'], moments['mu11']],
        [moments['mu11'], moments['mu02']]
    ]), area)

    _, svd_u, _ = cv2.SVDecomp(moments_matrix)

    center = np.array([mean_x, mean_y])
    tangent = svd_u[:, 0].flatten().copy()

    return center, tangent


class ContourInfo(object):
    def __init__(self, contour, rect, mask):

        self.contour = contour
        self.rect = rect
        self.mask = mask

        self.center, self.tangent = blob_mean_and_tangent(contour)

        self.angle = np.arctan2(self.tangent[1], self.tangent[0])

        clx = [self.proj_x(point) for point in contour]

        lxmin = min(clx)
        lxmax = max(clx)

        self.local_xrng = (lxmin, lxmax)

        self.point0 = self.center + self.tangent * lxmin
        self.point1 = self.center + self.tangent * lxmax

        self.pred = None
        self.succ = None

    def proj_x(self, point):
        return np.dot(self.tangent, point.flatten() - self.center)

    def local_overlap(self, other):
        xmin = self.proj_x(other.point0)
        xmax = self.proj_x(other.point1)
        return interval_measure_overlap(self.local_xrng, (xmin, xmax))


def generate_candidate_edge(cinfo_a, cinfo_b):

    # we want a left of b (so a's successor will be b and b's
    # predecessor will be a) make sure right endpoint of b is to the
    # right of left endpoint of a.
    if cinfo_a.point0[0] > cinfo_b.point1[0]:
        tmp = cinfo_a
        cinfo_a = cinfo_b
        cinfo_b = tmp

    x_overlap_a = cinfo_a.local_overlap(cinfo_b)
    x_overlap_b = cinfo_b.local_overlap(cinfo_a)

    overall_tangent = cinfo_b.center - cinfo_a.center
    overall_angle = np.arctan2(overall_tangent[1], overall_tangent[0])

    delta_angle = old_div(max(angle_dist(cinfo_a.angle, overall_angle),
                      angle_dist(cinfo_b.angle, overall_angle)) * 180, np.pi)

    # we want the largest overlap in x to be small
    x_overlap = max(x_overlap_a, x_overlap_b)

    dist = np.linalg.norm(cinfo_b.point0 - cinfo_a.point1)

    if (dist > EDGE_MAX_LENGTH or x_overlap > EDGE_MAX_OVERLAP or delta_angle > EDGE_MAX_ANGLE):
        return None
    else:
        score = dist + delta_angle * EDGE_ANGLE_COST
        return (score, cinfo_a, cinfo_b)


def make_tight_mask(contour, xmin, ymin, width, height):

    tight_mask = np.zeros((height, width), dtype=np.uint8)
    tight_contour = contour - np.array((xmin, ymin)).reshape((-1, 1, 2))

    cv2.drawContours(tight_mask, [tight_contour], 0,
                     (1, 1, 1), -1)

    return tight_mask


def get_contours(name, small, pagemask, masktype):

    mask = get_mask(name, small, pagemask, masktype)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)

    contours_out = []

    for contour in contours:

        rect = cv2.boundingRect(contour)
        xmin, ymin, width, height = rect

        if (width < TEXT_MIN_WIDTH or height < TEXT_MIN_HEIGHT or width < TEXT_MIN_ASPECT * height):
            continue

        tight_mask = make_tight_mask(contour, xmin, ymin, width, height)

        if tight_mask.sum(axis=0).max() > TEXT_MAX_THICKNESS:
            continue

        contours_out.append(ContourInfo(contour, rect, tight_mask))

    if DEBUG_LEVEL >= 2:
        visualize_contours(name, small, contours_out)

    return contours_out


def assemble_spans(name, small, pagemask, cinfo_list):

    # sort list
    cinfo_list = sorted(cinfo_list, key=lambda cinfo: cinfo.rect[1])

    # generate all candidate edges
    candidate_edges = []

    for i, cinfo_i in enumerate(cinfo_list):
        for j in range(i):
            # note e is of the form (score, left_cinfo, right_cinfo)
            edge = generate_candidate_edge(cinfo_i, cinfo_list[j])
            if edge is not None:
                candidate_edges.append(edge)

    # sort candidate edges by score (lower is better)
    candidate_edges.sort()

    # for each candidate edge
    for _, cinfo_a, cinfo_b in candidate_edges:
        # if left and right are unassigned, join them
        if cinfo_a.succ is None and cinfo_b.pred is None:
            cinfo_a.succ = cinfo_b
            cinfo_b.pred = cinfo_a

    # generate list of spans as output
    spans = []

    # until we have removed everything from the list
    while cinfo_list:

        # get the first on the list
        cinfo = cinfo_list[0]

        # keep following predecessors until none exists
        while cinfo.pred:
            cinfo = cinfo.pred

        # start a new span
        cur_span = []

        width = 0.0

        # follow successors til end of span
        while cinfo:
            # remove from list (sadly making this loop *also* O(n^2)
            cinfo_list.remove(cinfo)
            # add to span
            cur_span.append(cinfo)
            width += cinfo.local_xrng[1] - cinfo.local_xrng[0]
            # set successor
            cinfo = cinfo.succ

        # add if long enough
        if width > SPAN_MIN_WIDTH:
            spans.append(cur_span)

    if DEBUG_LEVEL >= 2:
        visualize_spans(name, small, pagemask, spans)

    return spans


def sample_spans(shape, spans):

    span_points = []

    for span in spans:

        contour_points = []

        for cinfo in span:

            yvals = np.arange(cinfo.mask.shape[0]).reshape((-1, 1))
            totals = (yvals * cinfo.mask).sum(axis=0)
            means = old_div(totals, cinfo.mask.sum(axis=0))

            xmin, ymin = cinfo.rect[:2]

            step = SPAN_PX_PER_STEP
            start = old_div(((len(means) - 1) % step), 2)

            contour_points += [(x + xmin, means[x] + ymin)
                               for x in range(start, len(means), step)]

        contour_points = np.array(contour_points,
                                  dtype=np.float32).reshape((-1, 1, 2))

        contour_points = pix2norm(shape, contour_points)

        span_points.append(contour_points)

    return span_points


def keypoints_from_samples(name, small, pagemask, page_outline,
                           span_points):

    all_evecs = np.array([[0.0, 0.0]])
    all_weights = 0

    for points in span_points:

        _, evec = cv2.PCACompute(points.reshape((-1, 2)),
                                 None, maxComponents=1)

        weight = np.linalg.norm(points[-1] - points[0])

        all_evecs += evec * weight
        all_weights += weight

    evec = old_div(all_evecs, all_weights)

    x_dir = evec.flatten()

    if x_dir[0] < 0:
        x_dir = -x_dir

    y_dir = np.array([-x_dir[1], x_dir[0]])

    pagecoords = cv2.convexHull(page_outline)
    pagecoords = pix2norm(pagemask.shape, pagecoords.reshape((-1, 1, 2)))
    pagecoords = pagecoords.reshape((-1, 2))

    px_coords = np.dot(pagecoords, x_dir)
    py_coords = np.dot(pagecoords, y_dir)

    px0 = px_coords.min()
    px1 = px_coords.max()

    py0 = py_coords.min()
    py1 = py_coords.max()

    p00 = px0 * x_dir + py0 * y_dir
    p10 = px1 * x_dir + py0 * y_dir
    p11 = px1 * x_dir + py1 * y_dir
    p01 = px0 * x_dir + py1 * y_dir

    corners = np.vstack((p00, p10, p11, p01)).reshape((-1, 1, 2))

    ycoords = []
    xcoords = []

    for points in span_points:
        pts = points.reshape((-1, 2))
        px_coords = np.dot(pts, x_dir)
        py_coords = np.dot(pts, y_dir)
        ycoords.append(py_coords.mean() - py0)
        xcoords.append(px_coords - px0)

    if DEBUG_LEVEL >= 2:
        visualize_span_points(name, small, span_points, corners)

    return corners, np.array(ycoords), xcoords


def visualize_contours(name, small, cinfo_list):

    regions = np.zeros_like(small)

    for j, cinfo in enumerate(cinfo_list):

        cv2.drawContours(regions, [cinfo.contour], 0,
                         CCOLORS[j % len(CCOLORS)], -1)

    mask = (regions.max(axis=2) != 0)

    display = small.copy()
    display[mask] = (old_div(display[mask], 2)) + (old_div(regions[mask], 2))

    for j, cinfo in enumerate(cinfo_list):
        color = CCOLORS[j % len(CCOLORS)]
        color = tuple([old_div(c, 4) for c in color])

        cv2.circle(display, fltp(cinfo.center), 3,
                   (255, 255, 255), 1, cv2.LINE_AA)

        cv2.line(display, fltp(cinfo.point0), fltp(cinfo.point1),
                 (255, 255, 255), 1, cv2.LINE_AA)

    debug_show(name, 1, 'contours', display)


def visualize_spans(name, small, pagemask, spans):

    regions = np.zeros_like(small)

    for i, span in enumerate(spans):
        contours = [cinfo.contour for cinfo in span]
        cv2.drawContours(regions, contours, -1,
                         CCOLORS[i * 3 % len(CCOLORS)], -1)

    mask = (regions.max(axis=2) != 0)

    display = small.copy()
    display[mask] = (old_div(display[mask], 2)) + (old_div(regions[mask], 2))
    display[pagemask == 0] //= 4

    debug_show(name, 2, 'spans', display)


def visualize_span_points(name, small, span_points, corners):

    display = small.copy()

    for i, points in enumerate(span_points):

        points = norm2pix(small.shape, points, False)

        mean, small_evec = cv2.PCACompute(points.reshape((-1, 2)),
                                          None,
                                          maxComponents=1)

        dps = np.dot(points.reshape((-1, 2)), small_evec.reshape((2, 1)))
        dpm = np.dot(mean.flatten(), small_evec.flatten())

        point0 = mean + small_evec * (dps.min() - dpm)
        point1 = mean + small_evec * (dps.max() - dpm)

        for point in points:
            cv2.circle(display, fltp(point), 3,
                       CCOLORS[i % len(CCOLORS)], -1, cv2.LINE_AA)

        cv2.line(display, fltp(point0), fltp(point1),
                 (255, 255, 255), 1, cv2.LINE_AA)

    cv2.polylines(display, [norm2pix(small.shape, corners, True)],
                  True, (255, 255, 255))

    debug_show(name, 3, 'span points', display)


def imgsize(img):
    height, width = img.shape[:2]
    return '{}x{}'.format(width, height)


def make_keypoint_index(span_counts):

    nspans = len(span_counts)
    npts = sum(span_counts)
    keypoint_index = np.zeros((npts + 1, 2), dtype=int)
    start = 1

    for i, count in enumerate(span_counts):
        end = start + count
        keypoint_index[start:start + end, 1] = 8 + i
        start = end

    keypoint_index[1:, 0] = np.arange(npts) + 8 + nspans

    return keypoint_index


def optimize_params(name, small, dstpoints, span_counts, params):

    keypoint_index = make_keypoint_index(span_counts)

    def objective(pvec):
        ppts = project_keypoints(pvec, keypoint_index)
        return np.sum((dstpoints - ppts)**2)

    print('  initial objective is', objective(params))

    if DEBUG_LEVEL >= 1:
        projpts = project_keypoints(params, keypoint_index)
        display = draw_correspondences(small, dstpoints, projpts)
        debug_show(name, 4, 'keypoints before', display)

    print('  optimizing', len(params), 'parameters...')
    start = datetime.datetime.now()
    res = scipy.optimize.minimize(objective, params,
                                  method='Powell')
    end = datetime.datetime.now()
    print('  optimization took', round((end - start).total_seconds(), 2), 'sec.')
    print('  final objective is', res.fun)
    params = res.x

    if DEBUG_LEVEL >= 1:
        projpts = project_keypoints(params, keypoint_index)
        display = draw_correspondences(small, dstpoints, projpts)
        debug_show(name, 5, 'keypoints after', display)

    return params


def get_page_dims(corners, rough_dims, params):

    dst_br = corners[2].flatten()

    dims = np.array(rough_dims)

    def objective(dims):
        proj_br = project_xy(dims, params)
        return np.sum((dst_br - proj_br.flatten())**2)

    res = scipy.optimize.minimize(objective, dims, method='Powell')
    dims = res.x

    print('  got page dims', dims[0], 'x', dims[1])

    return dims


def remap_image(name, img, small, page_dims, params, output_path):

    height = 0.5 * page_dims[1] * OUTPUT_ZOOM * img.shape[0]
    height = round_nearest_multiple(height, REMAP_DECIMATE)

    width = round_nearest_multiple(old_div(height * page_dims[0], page_dims[1]),
                                   REMAP_DECIMATE)

    print('  output will be {}x{}'.format(width, height))

    height_small = old_div(height, REMAP_DECIMATE)
    width_small = old_div(width, REMAP_DECIMATE)

    page_x_range = np.linspace(0, page_dims[0], width_small)
    page_y_range = np.linspace(0, page_dims[1], height_small)

    page_x_coords, page_y_coords = np.meshgrid(page_x_range, page_y_range)

    page_xy_coords = np.hstack((page_x_coords.flatten().reshape((-1, 1)),
                                page_y_coords.flatten().reshape((-1, 1))))

    page_xy_coords = page_xy_coords.astype(np.float32)

    image_points = project_xy(page_xy_coords, params)
    image_points = norm2pix(img.shape, image_points, False)

    image_x_coords = image_points[:, 0, 0].reshape(page_x_coords.shape)
    image_y_coords = image_points[:, 0, 1].reshape(page_y_coords.shape)

    image_x_coords = cv2.resize(image_x_coords, (width, height),
                                interpolation=cv2.INTER_CUBIC)

    image_y_coords = cv2.resize(image_y_coords, (width, height),
                                interpolation=cv2.INTER_CUBIC)

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    remapped = cv2.remap(img_gray, image_x_coords, image_y_coords,
                         cv2.INTER_CUBIC,
                         None, cv2.BORDER_REPLICATE)

    thresh = cv2.adaptiveThreshold(remapped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, ADAPTIVE_WINSZ, 25)
    # thresh = cv2.adaptiveThreshold(remapped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 4)

    pil_image = Image.fromarray(thresh)
    pil_image = pil_image.convert('1')

    threshfile = name + '_thresh.png'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_threshfile = os.path.join(output_path, threshfile)
    pil_image.save(output_threshfile, dpi=(OUTPUT_DPI, OUTPUT_DPI))

    if DEBUG_LEVEL >= 1:
        height = small.shape[0]
        width = int(round(height * float(thresh.shape[1]) / thresh.shape[0]))
        display = cv2.resize(thresh, (width, height),
                             interpolation=cv2.INTER_AREA)
        debug_show(name, 6, 'output', display)

    return output_threshfile


def page_dewarp(imgfile, output_path="threshes"):

    if DEBUG_LEVEL > 0 and DEBUG_OUTPUT != 'file':
        cv2.namedWindow(WINDOW_NAME)

    img = cv2.imread(imgfile)
    small = resize_to_screen(img)
    basename = os.path.basename(imgfile)
    name, _ = os.path.splitext(basename)

    print('loaded', basename, 'with size', imgsize(img), end=' ')
    print('and resized to', imgsize(small))

    if DEBUG_LEVEL >= 3:
        debug_show(name, 0.0, 'original', small)

    pagemask, page_outline = get_page_extents(small)

    cinfo_list = get_contours(name, small, pagemask, 'text')
    spans = assemble_spans(name, small, pagemask, cinfo_list)

    if len(spans) < 3:
        print('  detecting lines because only', len(spans), 'text spans')
        cinfo_list = get_contours(name, small, pagemask, 'line')
        spans2 = assemble_spans(name, small, pagemask, cinfo_list)
        if len(spans2) > len(spans):
            spans = spans2

    if len(spans) < 1:
        print('skipping', name, 'because only', len(spans), 'spans')
        return

    span_points = sample_spans(small.shape, spans)

    print('  got', len(spans), 'spans', end=' ')
    print('with', sum([len(pts) for pts in span_points]), 'points.')

    corners, ycoords, xcoords = keypoints_from_samples(name, small,
                                                        pagemask,
                                                        page_outline,
                                                        span_points)

    rough_dims, span_counts, params = get_default_params(corners,
                                                            ycoords, xcoords)

    dstpoints = np.vstack((corners[0].reshape((1, 1, 2)),) + tuple(span_points))

    params = optimize_params(name, small,
                                dstpoints,
                                span_counts, params)

    page_dims = get_page_dims(corners, rough_dims, params)

    outfile = remap_image(name, img, small, page_dims, params, output_path)

    return outfile

# CTPN SHIT STARTS HERE

def resize_image(img):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(600) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1200:
        im_scale = float(1200) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

    re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return re_im, (new_h / img_size[0], new_w / img_size[1])


def ctpn(imgfile):
    if imgfile.strip() == "":
        raise IOError
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

    output_img_file, txt_file = "", ""

    try:
        with tf.get_default_graph().as_default():
            input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
            input_im_info = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')

            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

            bbox_pred, cls_pred, cls_prob = model.model(input_image)

            variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
            saver = tf.train.Saver(variable_averages.variables_to_restore())

            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
                model_path = os.path.join(checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
                print('Restore from {}'.format(model_path))
                saver.restore(sess, model_path)

                start = time.time()
                try:
                    im = cv2.imread(imgfile)[:, :, ::-1]
                except:
                    print("Error reading image {}!".format(imgfile))
                    tf.reset_default_graph()
                    raise IOError

                img, (rh, rw) = resize_image(im)
                h, w, c = img.shape
                im_info = np.array([h, w, c]).reshape([1, 3])
                bbox_pred_val, cls_prob_val = sess.run([bbox_pred, cls_prob],
                                                    feed_dict={input_image: [img],
                                                            input_im_info: im_info})

                textsegs, _ = proposal_layer(cls_prob_val, bbox_pred_val, im_info)
                scores = textsegs[:, 0]
                textsegs = textsegs[:, 1:5]

                textdetector = TextDetector(DETECT_MODE='O')
                boxes = textdetector.detect(textsegs, scores[:, np.newaxis], img.shape[:2])
                boxes = np.array(boxes, dtype=np.int)

                cost_time = (time.time() - start)
                print("cost time: {:.2f}s".format(cost_time))

                for i, box in enumerate(boxes):
                    cv2.polylines(img, [box[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0),
                                thickness=2)
                img = cv2.resize(img, None, None, fx=1.0 / rh, fy=1.0 / rw, interpolation=cv2.INTER_LINEAR)
                output_split = os.path.splitext(os.path.basename(imgfile))
                output_name = output_split[0] + "_ctpn" + output_split[1]
                output_img_file = os.path.join(output_path, output_name)
                cv2.imwrite(output_img_file, img[:, :, ::-1])

                txt_file = os.path.join(output_path, os.path.splitext(os.path.basename(imgfile))[0]) + ".txt"
                with open(txt_file, "w") as f:
                    for i, box in enumerate(boxes):
                        line = ",".join(str(box[k]) for k in range(8))
                        line += "," + str(scores[i]) + "\r\n"
                        f.writelines(line)
    except:
        tf.reset_default_graph()
        traceback.print_exc()

    tf.reset_default_graph()
    return output_img_file, txt_file

# OCR SHIT STARTS HERE
def get_bounding_box(txt):
    annotation = txt
    with open(annotation, "r") as file1:
        bounding_boxes = file1.read()

    bounding_boxes = bounding_boxes.split('\n')[:-1]
    boxes = [i.split(',')[:-1] for i in bounding_boxes]

    new_boxes = []
    for box in boxes:
        new_box = []
        for i, each in enumerate(box):
            num = int(each)
            if i in [0, 1, 3, 6]:
                num -= 3
            else:
                num += 3
            new_box.append(num)
        new_boxes.append(new_box)
    new_boxes.sort(key=lambda x: x[1])

    return new_boxes

def clean_string(string):
    text = string.replace('INACTIVE INGREDIENTS:', '') # added
    text = text.replace('ACTIVE INGREDIENTS:', '') # added
    # text = text.split(':')[1] THIS IS WRONG
    
    pattern = "[\|\*\_\'\{}&]".format('"')
    regex = re.compile('\\\S+')
    
    text = re.sub(pattern, "", text)
    text = re.sub(",, ", ", ", text)
    text = re.sub(regex, " ", text)
    text = re.sub('\.', " ", text)
    text_tokens = word_tokenize(text)
    text_wo_sw = [w for w in text_tokens if w not in stopwords.words()]
    text = ' '.join(text_wo_sw)
    text = text.strip()

    return text

def string_to_list(text):
    pattern = "[\|\*\_\'\{}]".format('"')
    text = re.sub(pattern, "", text)
    split = [remove_water(x) for x in re.split("[,.]", text)]

    return split

def remove_water(string):
    water = ['WATER (AQUA)', 'AQUA', 'EAU', 'AQUA/WATER/EAU', 'AQUA / WATER / EAU',
             'PURIFIED WATER', 'DISTILLED WATER', 'D.I. WATER', 'AQUA (WATER)', 'AQUA (PURIFIED)']
    text = string.upper()
    if text in water:
        text = 'WATER'
    text = text.strip('  ')

    return text

def crop_line(img_path, box):
    img = cv2.imread(img_path)
    img, (rh, rw) = resize_image(img)
    # points for test.jpg
    cnt = np.array([
        [[box[0], box[1]]],
        [[box[2], box[3]]],
        [[box[4], box[5]]],
        [[box[6], box[7]]]
    ])
    # print("shape of cnt: {}".format(cnt.shape))
    rect = cv2.minAreaRect(cnt)
#     print("rect: {}".format(rect))

    # the order of the box points: bottom left, top left, top right,
    # bottom right
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # print("bounding box: {}".format(box))
    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    # get width and height of the detected rectangle
    width = int(rect[1][0])
    height = int(rect[1][1])
    angle = rect[2]

    src_pts = box.astype("float32")
    # coordinate of the points in box points after the rectangle has been
    # straightened
    dst_pts = np.array([[0, height + 2],
        [0, 0],
        [width, 0],
        [width, height + 2]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(img, M, (width, height))

    # cv2.imwrite("crop_img.jpg", warped)

    # cv2.waitKey(0)
    if angle < -45:
        warped = np.transpose(warped, (1, 0, 2))
        warped = warped[::-1]

#     cv2.imshow('croped', warped)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    return warped

def ocr(img, oem=3, psm=6):
    """
    @param img: The image to be OCR'd
    @param oem: for specifying the type of Tesseract engine( default=1 for LSTM OCR Engine)
    """
    config = ('-l eng --oem {oem} --psm {psm}'.format(oem=oem, psm=psm))
    try:
        text = pytesseract.image_to_string(img, config=config)
        return text
    except:
        return ""

class FuzzyDict(dict):
    "Provides a dictionary that performs fuzzy lookup"
    def __init__(self, items=None, cutoff=.6):
        """Construct a new FuzzyDict instance

        items is an dictionary to copy items from (optional)
        cutoff is the match ratio below which mathes should not be considered
        cutoff needs to be a float between 0 and 1 (where zero is no match
        and 1 is a perfect match)"""
        super(FuzzyDict, self).__init__()

        if items:
            self.update(items)
        self.cutoff = cutoff

        # short wrapper around some super (dict) methods
        self._dict_contains = lambda key: \
            super(FuzzyDict, self).__contains__(key)

        self._dict_getitem = lambda key: \
            super(FuzzyDict, self).__getitem__(key)

    def _search(self, lookfor, stop_on_first=False):
        """Returns the value whose key best matches lookfor

        if stop_on_first is True then the method returns as soon
        as it finds the first item
        """

        # if the item is in the dictionary then just return it
        if self._dict_contains(lookfor):
            return True, lookfor, self._dict_getitem(lookfor), 1

        # set up the fuzzy matching tool
        # ratio_calc = difflib.SequenceMatcher()
        # ratio_calc.set_seq1(lookfor)

        # test each key in the dictionary
        best_ratio = 0
        best_match = None
        best_key = None
        for key in self:

            # if the current key is not a string
            # then we just skip it
            if not isinstance(key, str):
                continue

            # we get an error here if the item to look for is not a
            # string - if it cannot be fuzzy matched and we are here
            # this it is defintely not in the dictionary
            try:
                # calculate the match value
                ratio = fuzz.ratio(lookfor, key) / 100
            except TypeError:
                break

            # if this is the best ratio so far - save it and the value
            if ratio > best_ratio:
                best_ratio = ratio
                best_key = key
                best_match = self._dict_getitem(key)

            if stop_on_first and ratio >= self.cutoff:
                break

        return (
            best_ratio >= self.cutoff,
            best_key,
            best_match,
            best_ratio)

    def __contains__(self, item):
        "Overides Dictionary __contains__ to use fuzzy matching"
        if self._search(item, True)[0]:
            return True
        else:
            return False

    def __getitem__(self, lookfor):
        "Overides Dictionary __getitem__ to use fuzzy matching"
        matched, key, item, ratio = self._search(lookfor)

        if not matched:
            raise KeyError(
                "'%s'. closest match: '%s' with ratio %.3f" % (str(lookfor), str(key), ratio))

        return item

def fuzzy_match_ingredients(ing_list, fuzdict):
    match_dict = {}
    for ing in tqdm(ing_list):
        if ing in match_dict.keys():
            continue
        upper_ing = ing.upper()
        if fuzdict.__contains__(upper_ing):
            match_dict[ing] = fuzdict[upper_ing]
        else:
            match_dict[ing] = 'unknown'

    return match_dict

def create_dict_english(df_inci, df_cosing):
    rating_inci = {}
    irritancy_inci = {}
    comedogenicity_inci = {}
    function_inci = {}
    qfacts_inci = {}
    desc_inci = {}
    
    desc_cosing = {}
    function_cosing = {}
    
    for idx, row in tqdm(df_inci.iterrows()):
        for name in row['ingredient_name'].split('/'):
            chem_name = name.strip()
            rating_inci[chem_name] = row['rating']
            irritancy_inci[chem_name] = row['irritancy']
            comedogenicity_inci[chem_name] = row['comedogenicity']
            function_inci[chem_name] = row['functions']
            qfacts_inci[chem_name] = row['quick_facts']
            desc_inci[chem_name] = row['description']
            
    for idx, row in tqdm(df_cosing.iterrows()):
        for name in row['ingredient_name'].split('/'):
            desc_cosing[name] = row['description']
            function_cosing[name] = row['functions']    
    
    return rating_inci, irritancy_inci, comedogenicity_inci, function_inci, qfacts_inci, desc_inci, desc_cosing, function_cosing

def lookup_all_english(ingredient_list, match_dict_inci, match_dict_cosing,
               df_inci, df_cosing, option=''):

    with open('pickles/eng_rating_inci.pickle', 'rb') as handle:
        rating_inci = pickle.load(handle)
    with open('pickles/eng_irritancy_inci.pickle', 'rb') as handle:
        irritancy_inci = pickle.load(handle)
    with open('pickles/eng_comedogenicity_inci.pickle', 'rb') as handle:
        comedogenicity_inci = pickle.load(handle)
    with open('pickles/eng_function_inci.pickle', 'rb') as handle:
        function_inci = pickle.load(handle)
    with open('pickles/eng_qfacts_inci.pickle', 'rb') as handle:
        qfacts_inci = pickle.load(handle)
    with open('pickles/eng_desc_inci.pickle', 'rb') as handle:
        desc_inci = pickle.load(handle)
    with open('pickles/eng_desc_cosing.pickle', 'rb') as handle:
        desc_cosing = pickle.load(handle)
    with open('pickles/eng_function_cosing.pickle', 'rb') as handle:
        function_cosing = pickle.load(handle)

    res = []

    for item in tqdm(ingredient_list):

        value = match_dict_inci[item]
        if value == 'unknown':
            key = match_dict_cosing.get(item, 'unknown')
            rating = 'No rating'
            irritancy = np.nan
            comedogenicity = np.nan
            functions = function_cosing.get(key, [])
            quickfacts = np.nan
            description = desc_cosing.get(key, [])

        else:
            key = match_dict_inci.get(item, 'unknown')
            rating = rating_inci.get(key, 'No rating')
            irritancy = irritancy_inci.get(key, np.nan)
            comedogenicity = comedogenicity_inci.get(key, np.nan)
            functions = function_inci.get(key, [])
            quickfacts = qfacts_inci.get(key, [])
            description = desc_inci.get(key, [])

        if key != 'unknown':
            if option == 'ingredient':
                res.append(key)
            elif option == 'rating':
                res.append(rating)
            elif option == 'irritancy':
                res.append(irritancy)
            elif option == 'comedogenicity':
                res.append(comedogenicity)
            elif option == 'functions':
                res.append(functions)
            elif option == 'quickfacts':
                res.append(quickfacts)
            elif option == 'description':
                res.append(description)
            else:
                res.extend([[key, functions, rating, irritancy, comedogenicity, quickfacts, description]])
            
    df_res = pd.DataFrame(res, columns=['Ingredient_name', 'Functions', 'Rating', 'Irritancy',
                                        'Comedogenicity', 'Quick_facts', 'Description'])
    
    return df_res

def lookup_all_vietnamese(ingredient_list, match_dict_cmd, match_dict_cosing,
               df_cmd, df_cosing, option=''):

    with open('pickles/vie_ratingscore_cmd.pickle', 'rb') as handle:
        ratingscore_cmd = pickle.load(handle)
    with open('pickles/vie_function_cmd.pickle', 'rb') as handle:
        function_cmd = pickle.load(handle)
    with open('pickles/vie_desc_cmd.pickle', 'rb') as handle:
        desc_cmd = pickle.load(handle)

    with open('pickles/eng_desc_cosing.pickle', 'rb') as handle:
        desc_cosing = pickle.load(handle)
    with open('pickles/eng_function_cosing.pickle', 'rb') as handle:
        function_cosing = pickle.load(handle)

    res = []

    for item in tqdm(ingredient_list):

        value = match_dict_cmd[item]

        if value == 'unknown':
            key = match_dict_cosing.get(item, 'unknown')
            rating_score = 'Chưa đánh giá'
            functions = function_cosing.get(key, [])
            description = desc_cosing.get(key, [])
        else:
            key = match_dict_cmd.get(item, 'unknown')
            rating_score = ratingscore_cmd.get(key, np.nan)
            functions = function_cmd.get(key, [])
            description = desc_cmd.get(key, [])

        if key != 'unknown':
            if option == 'ingredient':
                res.append(key)
            elif option == 'rating_score':
                res.append(rating_score)
            elif option == 'functions':
                res.append(functions)
            elif option == 'description':
                res.append(description)
            else:
                res.extend([[key, rating_score, functions, description]])

    df_res = pd.DataFrame(res, columns=['Ingredient_name', 'Rating', 'Functions', 'Description'])

    return df_res

# OK THE GREAT OCR FUNCTION
def ocr_everything(img_path, boundingtxt_file, inci_path, cmd_path, cosing_path, language, debug=False):
    boxes = get_bounding_box(boundingtxt_file)

    # doing OCR
    text = ''
    for box in boxes:
        cropped = crop_line(img_path, box)
        string = ocr(cropped)
        text = text + ' ' + str(string.strip('\n').strip('\x0c').strip())

    if debug:
        print(text)

    # Cleaning result from OCR
    text_result = clean_string(text)
    ing_list = string_to_list(text_result)

    if debug:
        print("-----")
        print(text_result)

    # Loading ingredient dataframe

    df_cosing = pd.read_csv(cosing_path) # '../Database/ingredient_cosing_37309.csv'
    # fd_cosing
    cosing_dict = {name.strip(): name.strip() for name in df_cosing['ingredient_name']}
    fd_cosing = FuzzyDict(cosing_dict, cutoff=.6)
    match_dict_cosing = fuzzy_match_ingredients(ing_list, fd_cosing)
    print('len fd cosing:', len(fd_cosing))
    # Input for later models: KNN and randomforest
    model_input = [[name for name in match_dict_cosing.values()]]

    # fd main
    if language == 'Vietnamese':
        df_cmd = pd.read_csv(cmd_path) # Vietnamese database
        cmd_dict = {name.strip(): name.strip() for name in df_cmd['ingredient_name']}
        fd_cmd = FuzzyDict(cmd_dict, cutoff=.7)
        match_dict_fuzzy = fuzzy_match_ingredients(ing_list, fd_cmd)
        print('len fd cmd:', len(fd_cmd))
    else:
        df_inci = pd.read_csv(inci_path) # '../Database/CALLMEDUY/ingredient_vietnamese_3818.csv'
        inci_dict = {name.strip(): name.strip() for name in df_inci['ingredient_name']}
        fd_inci = FuzzyDict(inci_dict, cutoff=.7)
        match_dict_fuzzy = fuzzy_match_ingredients(ing_list, fd_inci)
        print('len fd inci:', len(fd_inci))

    # Compare product ingredient list and database
    # match_dict = find_matching_ingredient(ing_list, rating, 0.55)

    if debug:
        print(match_dict_fuzzy)
        print(list(match_dict_fuzzy.values()))
        print(ing_list)

    if debug:
        print("length match_dict_fuzzy", len(match_dict_fuzzy))
        print("length match_dict_extra", len(match_dict_cosing))

    # Analyzing ingredient
    if language == 'Vietnamese':
        df_res = lookup_all_vietnamese(ing_list, match_dict_fuzzy, match_dict_cosing, df_cmd, df_cosing)

    else:
        df_res = lookup_all_english(ing_list, match_dict_fuzzy, match_dict_cosing, df_inci, df_cosing)

    if debug:
        print(df_res)

    return df_res, model_input

def transform(X): # X: list
    M = len(X)
    N = len(ingredient_idx)
    A = np.zeros((M, N), dtype=np.uint8)
    i = 0
    for ing_list in X:
        x = np.zeros(N, dtype=np.uint8)
        for ingredient in ing_list:
            # Get the index for each ingredient
            if ingredient in ingredient_idx.keys():
                idx = ingredient_idx[ingredient]
                x[idx] = 1
            else:
                pass

        A[i, :] = x
        i += 1
    input_rf = A[:, :1000]
    input_knn = A

    return input_rf, input_knn


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
