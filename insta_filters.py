from PIL import Image
from pylab import array, plot, show, axis, arange, figure, uint8
import numpy as np
import cv2 
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance

########Ginhaam#######
def Ginham(vignetteTest1):
    #contrast
    from PIL import Image, ImageEnhance
    def contrast(im):
        enhancer = ImageEnhance.Contrast(im)
        enhanced_im = enhancer.enhance(1.5)
        enhanced_im.save("try.png")
    contrastTest=contrast(vignetteTest1)
    #warm up
    from scipy.interpolate import UnivariateSpline
    spl = UnivariateSpline(range(4), range(4))
    spl(2)
    def create_LUT_8UC1(x, y):
        spl = UnivariateSpline(x, y)
        return spl(range(256))
    def warm_up( img_bgr_in ):
        incr_ch_lut = create_LUT_8UC1([0, 64, 128, 192, 256],
         [0, 70, 140, 210, 256])
        decr_ch_lut = create_LUT_8UC1([0, 64, 128, 192, 256],
         [0, 30,  80, 120, 192])
        c_b, c_g, c_r = cv2.split(img_bgr_in)
        c_r = cv2.LUT(c_r, incr_ch_lut).astype(np.uint8)
        c_b = cv2.LUT(c_b, decr_ch_lut).astype(np.uint8)
        img_bgr_warm = cv2.merge((c_b, c_g, c_r))
        c_b = cv2.LUT(c_b, incr_ch_lut).astype(np.uint8)
        # increase color saturation
        c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_bgr_warm,cv2.COLOR_BGR2HSV))
        c_s = cv2.LUT(c_s, decr_ch_lut).astype(np.uint8)
        img_bgr_warm = cv2.cvtColor(cv2.merge((c_h, c_s, c_v)),cv2.COLOR_HSV2BGR)
        return img_bgr_warm
    def contrast(im):
        enhancer = ImageEnhance.Contrast(im)
        enhanced_im = enhancer.enhance(0.95)
        enhanced_im.save("try2.png")

    
    #cv2.save("contrast",contrastTest)
    cont = cv2.imread("try.png")
    cont[:,:,1]=cont[:,:,1]*0.85
    #cv2.imshow("try",cont)
    cont=warm_up(cont)
    cv2.imwrite("cont.jpg",cont)
    vignetteTest1=Image.open("cont.jpg")
    contrastTest=contrast(vignetteTest1)
    tt=cv2.imread("try2.png")
    #cv2.imshow("com",cont)
    #cv2.imshow("com2",cont)
    cv2.imshow("final",tt)
    
#############Hefe###############
def Maven(image):
    #color saturation
    def saturation(image):
        hsvImg = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        #multiple by a factor to change the saturation
        hsvImg[...,1] = hsvImg[...,1]*1.05
        #multiple by a factor of less than 1 to reduce the brightness 
        # hsvImg[...,2] = hsvImg[...,2]*1.055
        image2=cv2.cvtColor(hsvImg,cv2.COLOR_HSV2BGR)
        return image2
    #warm up
    from scipy.interpolate import UnivariateSpline
    spl = UnivariateSpline(range(4), range(4))
    spl(2)
    def create_LUT_8UC1(x, y):
        spl = UnivariateSpline(x, y)
        return spl(range(256))
    def warm_up( img_bgr_in ):
        incr_ch_lut = create_LUT_8UC1([0, 64, 128, 192, 256],
         [0, 70, 140, 210, 256])
        decr_ch_lut = create_LUT_8UC1([0, 64, 128, 192, 256],
         [0, 30,  80, 120, 192])
        c_b, c_g, c_r = cv2.split(img_bgr_in)
        c_r = cv2.LUT(c_r, incr_ch_lut).astype(np.uint8)
        c_b = cv2.LUT(c_b, decr_ch_lut).astype(np.uint8)
        img_bgr_warm = cv2.merge((c_b, c_g, c_r))
        c_b = cv2.LUT(c_b, decr_ch_lut).astype(np.uint8)
        # increase color saturation
        c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_bgr_warm,cv2.COLOR_BGR2HSV))
        c_s = cv2.LUT(c_s, incr_ch_lut).astype(np.uint8)
        img_bgr_warm = cv2.cvtColor(cv2.merge((c_h, c_s, c_v)),cv2.COLOR_HSV2BGR)
        return img_bgr_warm
    #brightness
    from pylab import array, plot, show, axis, arange, figure, uint8
    def bright(image):
        maxIntensity = 255.0 # depends on dtype of image data
        x = arange(maxIntensity) 
        # Parameters for manipulating image data
        phi = 1
        theta = 1
        newImage0 = (maxIntensity/phi)*(image/(maxIntensity/theta))**0.8
        newImage0 = array(newImage0,dtype=uint8)
        return newImage0
    #contrast
    from PIL import Image, ImageEnhance
    def contrast(im):
        enhancer = ImageEnhance.Contrast(im)
        enhanced_im = enhancer.enhance(1.5)
        enhanced_im.save("enhanced12W.png")
    #color saturation
    def saturation(image):
        hsvImg = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        #multiple by a factor to change the saturation
        hsvImg[...,1] = hsvImg[...,1]*0.8
        #multiple by a factor of less than 1 to reduce the brightness 
        hsvImg[...,2] = hsvImg[...,2]*0.85
        image2=cv2.cvtColor(hsvImg,cv2.COLOR_HSV2BGR)
        return image2
    saturationTest = saturation(image)
    #cv2.imshow("test",saturationTest)
    warm_upTest = warm_up(saturationTest)
   
    #cv2.imshow("test3",warm_upTest)
    cv2.imwrite("warmup.jpg",warm_upTest)
    result = bright(warm_upTest)
   # result[:,:,1]=result[:,:,1]*0.75
    #cv2.imshow("hope",result)
    cv2.imwrite("plz.jpg",result)
    vignetteTest1=Image.open("plz.jpg")
    contrastTest=contrast(vignetteTest1)
    #cv2.save("contrast",contrastTest)
    cont = cv2.imread("enhanced12W.png")
    #res=np.hstack((cont,brig))
    cv2.imshow("final",result)
    
###########InkWell##############
def InkWell(img):
    def black_white(img):
        equ = cv2.equalizeHist(img)
        return equ
    def bright(image):
        maxIntensity = 255.0 # depends on dtype of image data
        x = arange(maxIntensity) 
        # Parameters for manipulating image data
        phi = 1
        theta = 1
        newImage0 = (maxIntensity/phi)*(image/(maxIntensity/theta))**1.5
        newImage0 = array(newImage0,dtype=uint8)
        return newImage0
    equ=black_white(img)
   # cv2.imshow("inkwell",equ)
    out=bright(equ)
    cv2.imshow("out2",out)
    #cv2.imwrite("out.jpg",out)
    
##########Nashville###################
def nashville(img):
  #warm up
  from scipy.interpolate import UnivariateSpline
  spl = UnivariateSpline(range(4), range(4))
  spl(2)
  def create_LUT_8UC1( x, y):
    spl = UnivariateSpline(x, y)
    return spl(range(256))
  def warm_up( img_bgr_in ):
    incr_ch_lut = create_LUT_8UC1([5, 64, 128, 192, 256],
    [5, 70, 140, 210, 256])
    decr_ch_lut = create_LUT_8UC1([5, 64, 128, 192, 256],
    [5, 30, 80, 120, 192])
    c_b, c_g, c_r = cv2.split(img_bgr_in)
    c_r = cv2.LUT(c_r, incr_ch_lut).astype(np.uint8)
    c_b = cv2.LUT(c_b, incr_ch_lut).astype(np.uint8)
    c_b = cv2.LUT(c_b, incr_ch_lut).astype(np.uint8)
    img_bgr_warm = cv2.merge((c_b, c_g, c_r))
    c_b = cv2.LUT(c_b, decr_ch_lut).astype(np.uint8)
    # increase color saturation
    c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_bgr_warm,cv2.COLOR_BGR2HSV))
    c_s = cv2.LUT(c_s, incr_ch_lut).astype(np.uint8)
    img_bgr_warm = cv2.cvtColor(cv2.merge((c_h, c_s, c_v)),cv2.COLOR_HSV2BGR)
    return img_bgr_warm
  #color saturation
  def saturation(image):
    hsvImg = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    #multiple by a factor to change the saturation
    hsvImg[...,1] = hsvImg[...,1]*0.991
    #multiple by a factor of less than 1 to reduce the brightness 
    hsvImg[...,2] = hsvImg[...,2]*1
    image2=cv2.cvtColor(hsvImg,cv2.COLOR_HSV2BGR)
    return image2
  #brightness
  from pylab import array, plot, show, axis, arange, figure, uint8
  def bright(image):
    maxIntensity = 255.0 # depends on dtype of image data
    x = arange(maxIntensity) 
    # Parameters for manipulating image data
    phi = 1
    theta = 1
    newImage0 = (maxIntensity/phi)*(image/(maxIntensity/theta))**0.72
    newImage0 = array(newImage0,dtype=uint8)
    return newImage0
  #contrast
  from PIL import Image, ImageEnhance
  def contrast(im):
    enhancer = ImageEnhance.Contrast(im)
    enhanced_im = enhancer.enhance(0.85)
    enhanced_im.save("enhanced12W.png")
  img[:,:,1]=img[:,:,1]*0.52
  img[:,:,0]=img[:,:,0]*0.9
  img[:,:,2]=img[:,:,2]*1
  #img[255, 153, 255]=img[255, 153, 255]*1
  #cv2.imshow("pink",img)
  ttt=warm_up(img)
  i=bright(ttt)
  cv2.imshow("bright",i)
  #cv2.imwrite("ll.jpg",i)
  
#############clarendon#############
def clarendon (im):
    def contrast(im):
        enhancer = ImageEnhance.Contrast(im)
        enhanced_im = enhancer.enhance(1.15)
        enhanced_im.save("enhanced.sample812.png")
    contrast(im)
    img=cv2.imread("enhanced.sample812.png")
    img[:,:,0]=img[:,:,0]*0.8
    cv2.imshow("result",img)
###################################
Genral_image = 0
img = cv2.imread('test.jpg',0)     ##InkWell 
im = Image.open("test.jpg")        ##Ginham, clarendon
image = cv2.imread("test.jpg")     ##nashville
image2 = cv2.imread('test.jpg',1)  ##Maven

def insta_like(image,instafilter):
    if instafilter == "Ginham":
        Ginham(image)
    elif instafilter == "Maven":
        Maven(image)
    elif instafilter == "InkWell":
        InkWell(image)
    elif instafilter == "nashville":
        nashville(image)
    elif instafilter == "clarendon":
        clarendon(image)

print("Enter InstaFilter choose from(InkWell, Ginham, clarendon, Maven, nashville):")
x = input()
if x == "Ginham":
   General_image = im
elif x == "Maven":
   General_image = image2
elif x == "InkWell":
   General_image = img
elif x == "nashville":
   General_image = image
elif x == "clarendon":
   General_image = im
insta_like(General_image,x)
