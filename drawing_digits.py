from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import time

from sklearn.datasets import fetch_mldata

from Tkinter import *
import Image, ImageDraw
import cPickle

USE_MNIST = True
USE_PCA = True

if USE_MNIST:
  digits = fetch_mldata('MNIST original') # 28 x 28
  im_len = 28
  sub_len = int( 256 / im_len )
else:
  digits = load_digits() # 8 x 8
  im_len = 8
  sub_len = int( 256 / im_len )

X = digits.data
y = digits.target

#print ( X[0] )
#exit()

#pca = PCA()
#clf = svm.SVC()
#clf = svm.SVC(gamma=0.001, C=100)
#clf = SGDClassifier()
#clf = svm.SVC(kernel='rbf')
USE_PCA = False
clf = MultinomialNB()
"""
# 0.971 F1 score
if USE_PCA:
  pca = PCA( copy=True, n_components=24, whiten=False )
clf = KNeighborsClassifier(algorithm='auto', leaf_size=54, 
                           metric='minkowski',
                           n_neighbors=3, p=2, weights='distance')
"""
"""
# 0.973 F1 score
if USE_PCA:
  pca = PCA( copy=True, n_components=52, whiten=False )
clf = KNeighborsClassifier(algorithm='kd_tree', leaf_size=71, 
                           metric='minkowski',
                           n_neighbors=7, p=2, weights='uniform')
"""
"""
# 0.968 F1 score
USE_PCA=False
clf = ExtraTreesClassifier(bootstrap=False, compute_importances=None,
                           criterion='gini', max_depth=None,
                           max_features='sqrt',min_density=None,
                           min_samples_leaf=3.0,
                           min_samples_split=7.0,n_estimators=50, n_jobs=1,
                           oob_score=False, random_state=2, verbose=False)
"""
"""
# 0.981 F1 score
if USE_PCA:
  pca = PCA( copy=True, n_components=36, whiten=False )
clf = SVC( C=298572.771987, cache_size=1000.0, class_weight=None,
           coef0=0.0, degree=3, gamma=0.0, kernel='rbf', max_iter=77090822,
           probability=False,random_state=3, shrinking=False,
           tol=0.000220022502934, verbose=False )
"""

if USE_PCA:
  pca.fit(X)
  Xp = pca.transform(X)
  clf.fit(Xp,y)
  print( clf.score(Xp,y))
else:
  clf.fit(X,y)
  print( clf.score(X,y))

paint_radius = 12#8

def draw_on_canvas( event ):
  x1, y1 = ( event.x - paint_radius ), ( event.y - paint_radius )
  x2, y2 = ( event.x + paint_radius ), ( event.y + paint_radius )
  w.create_oval( x1, y1, x2, y2, fill='white', outline='white' )
  draw.ellipse( ( x1, y1, x2, y2 ), fill='white', outline='white' )
  pixels = image.load()
  drawn_image = convert_image()
  flat = np.reshape(drawn_image, (im_len*im_len))
  if USE_PCA:
    flat = pca.transform( flat )
  pred = clf.predict( flat )
  print( pred )
  label.config( text=str(pred) )

def clear_canvas():
  w.delete( ALL )
  draw.rectangle( ( 0, 0, 255, 255 ), fill='black', outline='black' )

def convert_image():
  pixels = image.load()
  converted_image = np.zeros((im_len,im_len))
  for i in xrange(im_len):
    for j in xrange(im_len):
      val = 0
      for x in xrange( sub_len ):
        for y in xrange( sub_len ):
          val += pixels[i*sub_len+x,j*sub_len+y]
      if USE_MNIST:
        converted_image[j,i] = round( val / ( sub_len * sub_len ), 0 )
      else:
        converted_image[j,i] = round( val * 16 / 256 / 1024, 0 )

  return converted_image
                               
tkroot = Tk()
w = Canvas( tkroot, width=256, height=256 )
w.config( bg='black' )
w.pack( side=BOTTOM )

label = Label( tkroot, text="Wheee" )
labelfont = ('courier', 48, 'bold')                
label.config( font=labelfont )
label.pack( side=RIGHT )
clear = Button( tkroot, text="Clear", command=clear_canvas )
clear.pack( side=TOP )

image = Image.new("L", (256, 256), 0)
draw = ImageDraw.Draw( image )

w.bind( '<B1-Motion>', draw_on_canvas )             

w.focus()                                     
tkroot.title( 'Digit Recognition' )
tkroot.mainloop()
