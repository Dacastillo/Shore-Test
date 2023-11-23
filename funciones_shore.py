import scipy.io
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from matplotlib.artist import Artist
import matplotlib.image as mpimg
import numpy as np
import numpy.matlib
from funciones_shore import *
from os import path
import pandas as pd
from scipy.optimize import curve_fit
from scipy import interpolate
from sklearn.metrics import mean_squared_error
from skimage.measure import profile_line
from skimage.measure import points_in_poly
from skimage.measure import find_contours
from skimage.filters import threshold_otsu
from scipy import stats
import time
import os, sys
import pickle
import warnings
warnings.filterwarnings("ignore")
# sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import sklearn
if sklearn.__version__[:4] == '0.20':
    from sklearn.externals import joblib
else:
    import joblib
from py import SDS_download, SDS_preprocess, SDS_shoreline, SDS_tools, SDS_classify

def dist(x, y):
    """
    Return the distance between two points.
    """
    d = x - y
    return np.sqrt(np.dot(d, d))


def dist_point_to_segment(p, s0, s1):
    """
    Get the distance of a point to a segment.
      *p*, *s0*, *s1* are *xy* sequences
    This algorithm from
    http://geomalgorithms.com/a02-_lines.html
    """
    v = s1 - s0
    w = p - s0
    c1 = np.dot(w, v)
    if c1 <= 0:
        return dist(p, s0)
    c2 = np.dot(v, v)
    if c2 <= c1:
        return dist(p, s1)
    b = c1 / c2
    pb = s0 + b * v
    return dist(p, pb)


class PolygonInteractor:
    """
    A polygon editor.

    Key-bindings

      't' toggle vertex markers on and off.  When vertex markers are on,
          you can move them, delete them

      'd' delete the vertex under point

      'i' insert a vertex at point.  You must be within epsilon of the
          line connecting two existing vertices

    """

    showverts = True
    epsilon = 5  # max pixel distance to count as a vertex hit

    def __init__(self, ax, poly):
        if poly.figure is None:
            raise RuntimeError('You must first add the polygon to a figure '
                               'or canvas before defining the interactor')
        self.ax = ax
        canvas = poly.figure.canvas
        self.poly = poly

        x, y = zip(*self.poly.xy)
        self.line = Line2D(x, y,
                           marker='o', markerfacecolor='r',
                           animated=True)
        self.ax.add_line(self.line)

        self.cid = self.poly.add_callback(self.poly_changed)
        self._ind = None  # the active vert

        canvas.mpl_connect('draw_event', self.on_draw)
        canvas.mpl_connect('button_press_event', self.on_button_press)
        canvas.mpl_connect('key_press_event', self.on_key_press)
        canvas.mpl_connect('button_release_event', self.on_button_release)
        canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas = canvas

    def on_draw(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        # do not need to blit here, this will fire before the screen is
        # updated

    def poly_changed(self, poly):
        """This method is called whenever the pathpatch object is called."""
        # only copy the artist props to the line (except visibility)
        vis = self.line.get_visible()
        Artist.update_from(self.line, poly)
        self.line.set_visible(vis)  # don't use the poly visibility state

    def get_ind_under_point(self, event):
        """
        Return the index of the point closest to the event position or *None*
        if no point is within ``self.epsilon`` to the event position.
        """
        # display coords
        xy = np.asarray(self.poly.xy)
        xyt = self.poly.get_transform().transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.hypot(xt - event.x, yt - event.y)
        indseq, = np.nonzero(d == d.min())
        ind = indseq[0]

        if d[ind] >= self.epsilon:
            ind = None

        return ind

    def on_button_press(self, event):
        """Callback for mouse button presses."""
        if not self.showverts:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self._ind = self.get_ind_under_point(event)

    def on_button_release(self, event):
        """Callback for mouse button releases."""
        if not self.showverts:
            return
        if event.button != 1:
            return
        self._ind = None

    def on_key_press(self, event):
        """Callback for key presses."""
        if not event.inaxes:
            return
        if event.key == 't':
            self.showverts = not self.showverts
            self.line.set_visible(self.showverts)
            if not self.showverts:
                self._ind = None
        elif event.key == 'd':
            ind = self.get_ind_under_point(event)
            if ind is not None:
                self.poly.xy = np.delete(self.poly.xy,
                                         ind, axis=0)
                self.line.set_data(zip(*self.poly.xy))
        elif event.key == 'i':
            xys = self.poly.get_transform().transform(self.poly.xy)
            p = event.x, event.y  # display coords
            for i in range(len(xys) - 1):
                s0 = xys[i]
                s1 = xys[i + 1]
                d = dist_point_to_segment(p, s0, s1)
                if d <= self.epsilon:
                    self.poly.xy = np.insert(
                        self.poly.xy, i+1,
                        [event.xdata, event.ydata],
                        axis=0)
                    self.line.set_data(zip(*self.poly.xy))
                    break
        if self.line.stale:
            self.canvas.draw_idle()

    def on_mouse_move(self, event):
        """Callback for mouse movements."""
        if not self.showverts:
            return
        if self._ind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        x, y = event.xdata, event.ydata

        self.poly.xy[self._ind] = x, y
        if self._ind == 0:
            self.poly.xy[-1] = x, y
        elif self._ind == len(self.poly.xy) - 1:
            self.poly.xy[0] = x, y
        self.line.set_data(zip(*self.poly.xy))

        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)
        
def TESTFITFUNC(xyz, beta0, beta1, beta2, beta3, beta4, beta5):
    #note: scipy.optimize.curve_fit required the parameters to fit as separate arguments in
    #the function definition. This is why beta1, beta2 etc... are written rather than just beta.
    mat = scipy.io.loadmat('data/RectifyImagePython.mat')
    
    fx = mat['fx'][0,0].astype(float)
    #note: it was found that -fy when data type is unit 16 returned a very unusul result.
    #so it is impotant these are dtype: float.
    fy = mat['fy'][0,0].astype(float)
    c0U = mat['c0U'][0,0].astype(float)
    c0V = mat['c0V'][0,0].astype(float)

    K = np.array([[fx, 0, c0U],[0, -fy, c0V],[0, 0, 1]]).astype(float)

    R = np.zeros((3, 3))

    def angles2R(a, t, s):
        R[0,0] = np.cos(a) * np.cos(s) + np.sin(a) * np.cos(t) * np.sin(s)
        R[0,1] = -np.cos(s) * np.sin(a) + np.sin(s) * np.cos(t) * np.cos(a)
        R[0,2] = np.sin(s) * np.sin(t)
        R[1,0] = -np.sin(s) * np.cos(a) + np.cos(s) * np.cos(t) * np.sin(a)
        R[1,1] = np.sin(s) * np.sin(a) + np.cos(s) * np.cos(t) * np.cos(a)
        R[1,2] = np.cos(s) * np.sin(t)
        R[2,0] = np.sin(t) * np.sin(a)
        R[2,1] = np.sin(t) * np.cos(a)
        R[2,2] = -np.cos(t)
    
    angles2R(beta3, beta4, beta5)

    I = np.eye(3)
    C = np.array([beta0, beta1, beta2]).astype(float)
    #to use np.hstack in the next cell, the arrays I and C both need to be 2D:
    C.shape = (3,1)

    IC = np.hstack((I,-C))

    P = np.matmul(np.matmul(K,R),IC)
    #note when comparing the output to P in MATLAB, the 1st and 3rd entries in the bottom
    #row are too small to show up in the MATLAB way of displaying the data. The results are
    #the same

    P = P/P[2,3]

    #note: instead of np.tranpose, an alternative is to place ".T" after the object to be transposed.
    #I keep np.transpose since this makes is more obvious to someone analysing the code.
    UV = np.matmul(P,np.vstack((np.transpose(xyz), np.ones((1, len(xyz)), dtype = float))))

    UV = UV/np.matlib.repmat(UV[2,:],3,1)
    UV = np.transpose(np.concatenate((UV[0,:], UV[1,:])))
    return UV
    
def onScreen(U, V, Umax, Vmax):
    Umin = 1
    Vmin = 1

    #Column of zeros, same length as UV(from the grid) (ie one for each coord set)
    yesNo = np.zeros((len(U),1))
    #Gives at 1 for all the UV coords which have existing corresponding pixel values from the oblique image
    on = np.where((U>=Umin) & (U<=Umax) & (V>=Vmin) & (V<=Vmax)) [0]
    yesNo[on] = 1
    return yesNo
