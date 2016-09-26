{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "  "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<hr style=\"height:3px;border:none;color:#333;background-color:#333;\" />\n",
    "<img style=\" float:right; display:inline\" src=\"http://opencloud.utsa.edu/wp-content/themes/utsa-oci/images/logo.png\"/>\n",
    "\n",
    "### **University of Texas at San Antonio** \n",
    "<br/>\n",
    "<br/>\n",
    "<span style=\"color:#000; font-family: 'Bebas Neue'; font-size: 2.5em;\"> **Open Cloud Institute** </span>\n",
    "\n",
    "<hr style=\"height:3px;border:none;color:#333;background-color:#333;\" />"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": true
   },
   "source": [
    "### Machine Learning/BigData EE-6973-001-Fall-2016\n",
    "\n",
    "<br/>\n",
    "<span style=\"color:#000; font-family: 'Bebas Neue'; font-size: 1.5em;\"> **Paul Rad, Ph.D.** </span>  \n",
    "\n",
    "\n",
    "<span style=\"color:#000; font-family: 'Bebas Neue'; font-size: 1.5em;\"> **Ali Miraftab, Research Fellow** </span>\n",
    "  \n",
    "\n",
    "<hr style=\"height:1.5px;border:none;color:#333;background-color:#333;\" />\n",
    "<hr style=\"height:1.5px;border:none;color:#333;background-color:#333;\" />"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<span style=\"color:#000; font-family: 'Bebas Neue'; font-size: 2em;\"> **Face Recognition Using a Hybrid ResNet-ConvNet Neural Network** </span>  \n",
    "<br/>\n",
    "<br/>\n",
    "<span style=\"color:#000; font-family: 'Bebas Neue'; font-size: 1.6em;\"> Ali Miraftab, Paul Rad </span>  \n",
    "<span style=\"color:#000; font-family: 'Bebas Neue'; font-size: 1.4em;\"> *Open Cloud Institute, University of Texas at San Antonio, San Antonio, Texas, USA* </span>  \n",
    "<span style=\"color:#000; font-family: 'Bebas Neue'; font-size: 1.4em;\"> {Ali.Miraftab, Paul.Rad}@utsa.edu </span>\n",
    "<br/>\n",
    "<br/>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<span style=\"color:#000; font-family: 'Bebas Neue'; font-size: 1.5em;\"> **Dataset:** </span> <span style=\"color:#000; font-family: 'Bebas Neue'; font-size: 1.3em;\"> The image data can be found in [http://www.cs.cmu.edu/afs/][1]. This directory contains 20 subdirectories, one for each person, named by userid.  Each of these directories contains several different face images of the same person. </span> \n",
    "\n",
    "[1]: http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-8/faceimages/faces/"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<span style=\"color:#000; font-family: 'Bebas Neue'; font-size: 1.5em;\"> **Outcome:** </span> <span style=\"color:#000; font-family: 'Bebas Neue'; font-size: 1.3em;\"> Applying resnet and convnet to identify the direction of the face </span>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<span style=\"color:#000; font-family: 'Bebas Neue'; font-size: 1.5em;\"> **Project Definition:** </span> <span style=\"color:#000; font-family: 'Bebas Neue'; font-size: 1.3em;\"> Using deep neural network models, ResNet and Convnet, classification of camera images of faces of various people in various poses is sudied. The dataset includes images of 20 different people, approximately 32 images per person, varying the person's expression (happy, sad, angry, neutral), the direction in which they are looking (left, right, straight ahead, up), and whether or not they were wearing sunglassess. </span>\n",
    "\n",
    "<span style=\"color:#000; font-family: 'Bebas Neue'; font-size: 1.3em;\"> There is also variation in the background behind the person, the clothing worn by the person, and theposition of the person's face within the image. I total. 624 greyscale images were collected, each whithin a resolutoin of 120by128, with each image pixel described by a greyscale intensity value between 0 (black) and 255 (white).</span>\n",
    "\n",
    "\n",
    "<span style=\"color:#000; font-family: 'Bebas Neue'; font-size: 1.3em;\"> A variety of target function can be learn from this image data. For example, given an image as input we could train a model to output the identity of the person, the direction in which the person is facing, the gender of the person, whether or not they are wearning sunglasses, etc. All of this target functions can be learned to high accuracy from this image data. In this course research we consider the particular task: learning the direction in which the person is facing (to their left, right, straight ahead, or upward)[1]. </span>\n",
    "\n",
    "[1]: Mitchell, T. M. (1997). Machine Learning. New York: McGraw-Hill."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "\n",
    "\n",
    "<div style=\"width:830; background-color:white; height:220px; overflow:scroll; overflow-x: scroll;overflow-y: hidden;\">\n",
    "\n",
    "<img style=\" float:left; display:inline\" src=\"\" width=\"160\" height=\"90\"/>\n",
    "\n",
    "<img style=\" float:left; display:inline\" src=\"https://archive.ics.uci.edu/ml/machine-learning-databases/faces-mld/at33_left_happy_sunglasses.jpg\" width=\"160\" height=\"90\"/>\n",
    "\n",
    "<img style=\" float:left; display:inline\" src=\"https://archive.ics.uci.edu/ml/machine-learning-databases/faces-mld/an2i_straight_neutral_open.jpg\" width=\"160\" height=\"90\"/>\n",
    "\n",
    "<img style=\" float:left; display:inline\" src=\"https://archive.ics.uci.edu/ml/machine-learning-databases/faces-mld/ch4f_up_angry_sunglasses.jpg\" width=\"160\" height=\"90\"/>\n",
    "\n",
    "<img style=\" float:left; display:inline\" src=\"https://archive.ics.uci.edu/ml/machine-learning-databases/faces-mld/boland_right_sad_open.jpg\" width=\"160\" height=\"90\"/>\n",
    "\n",
    "<img style=\" float:left; display:inline\" src=\"http://jmvidal.cse.sc.edu/talks/ann/mitchell-straight-happy-sunglasses-4.png\" width=\"160\" height=\"90\"/>\n",
    "\n",
    "</div>"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [Root]",
   "language": "python",
   "name": "Python [Root]"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.5.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
