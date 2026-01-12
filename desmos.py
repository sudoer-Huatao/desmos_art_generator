import cv2
import numpy as np

"""
HOW TO USE THIS SCRIPT:

Step 1. Get a image and drag it onto the Desktop. Name it test.jpg or change the path variable below.
Step 2. Run this script. It will generate a text file on your Desktop named desmos_equations.txt.
Step 3. Open desmos_equations.txt, copy all the equations, and paste them into Desmos (https://www.desmos.com/calculator).


NOTES:

1. GRAPHING_INACCURACY_VALUE controls the accuracy of the approximation.
The lower, the more accurate but the more equations.

2. fhand is the output file where the Desmos equations will be written. Edit the USERNAME to your
macOS username.

3. path is the input image path. Edit the USERNAME to your macOS username.

"""

GRAPHING_INACCURACY_VALUE = 0.002  # adjust this to change the accuracy of the approximation. The lower, the more accurate but the more equations.
fhand = open("/Users/USERNAME/Desktop/desmos_equations.txt", "w")
path = "/Users/USERNAME/Desktop/test.jpg"


def map_point(pt, w, h, scale):
    x, y = pt
    mx = (x - w / 2.0) * (2.0 * scale / w)
    my = (h / 2.0 - y) * (2.0 * scale / h)
    return mx, my

def generate_desmos_equations(path, max_contours=200, scale=10.0, approx_epsilon=2.0, min_seg_len=0.01):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    blur = cv2.GaussianBlur(img, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    eqs = []
    for ci, cnt in enumerate(contours):
        if ci >= max_contours: break
        peri = GRAPHING_INACCURACY_VALUE*cv2.arcLength(cnt, False)
        eps = max(1.0, approx_epsilon * (peri / max(w,h)))
        poly = cv2.approxPolyDP(cnt, eps, False)
        pts = [tuple(p[0]) for p in poly]
        if len(pts) < 2:
            continue
        # ensure closed by connecting last->first if far enough
        closed = True
        for i in range(len(pts)):
            p0 = pts[i]
            p1 = pts[(i+1) % len(pts)]
            a = np.hypot(p1[0]-p0[0], p1[1]-p0[1])
            if a < 1e-6:
                closed = False
                break
        seg_count = len(pts) if closed else len(pts)-1
        for i in range(seg_count):
            p0 = pts[i]
            p1 = pts[(i+1) % len(pts)]
            x0, y0 = map_point(p0, w, h, scale)
            x1, y1 = map_point(p1, w, h, scale)
            dx = x1 - x0
            dy = y1 - y0
            if np.hypot(dx, dy) < min_seg_len:
                continue
            # Desmos parametric segment: (x0 + dx t, y0 + dy t) {0<=t<=1}
            eq = "({:.4f} + {:.4f} t, {:.4f} + {:.4f} t)".format(x0, dx, y0, dy)
            eqs.append(eq)
    return eqs

if __name__ == "__main__":
    for line in generate_desmos_equations(path):
        fhand.write(line + "\n")


# Created by Huatao