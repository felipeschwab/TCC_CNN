import os.path
import natsort
import random
import itertools
import numpy as np
import cv2
import shutil
import glob,os.path

from matplotlib import pyplot as plt

DIR_IMAGES = "Images"
VIEWS_IMAGES = ["Inferior","Superior"]

BORDER = 25
SIZE = 250

THRESHOLD_LOWER = np.array([0,0,0])
THRESHOLD_UPPER = np.array([250,250,250])

N_NEW_IMAGES = 10
GENERATE_IMAGES = 1

def run():

    # Checa se os diretórios foram corretamente divididos pelo usuário
    filesDepth3 = glob.glob('Images/*/*/*')
    dirsDepth3 = list(filter(lambda f: os.path.isdir(f), filesDepth3))
    
    if len(dirsDepth3) > 0:
        print("ERRO: Os subfolders da pasta images devem ter uma profundidade maxima de 2.")
        print("As seguintes pasta não estão seguindo este critério: ")
        for depth in dirsDepth3:
            print(depth)
        #end for
        print("Corrigir antes de prosseguir!!!")
        return
    #end if
    
    filesDepth2 = glob.glob('Images/*/*')
    dirsDepth2 = list(filter(lambda f: os.path.isdir(f), filesDepth2))
    
    for views in dirsDepth2:
        if views.split("\\")[-1] not in VIEWS_IMAGES:
            print("ERRO: A vista %s do grupo %s não é suportada por este programa!!"%(views.split("\\")[-1],views.split("\\")[-2]))
            print("Corrigir de prosseguir!!!")
            return
        #end if
    #end for
    
    if os.path.isdir("training_images"):
        shutil.rmtree('training_images', ignore_errors=True)
    #end if
    
    #Cria os diretórios para cada grupo e os seus subdiretórios com as vistas
    for dirToCreate in filesDepth2:
        os.makedirs("training_images\\%s"%dirToCreate[7:])
    
    for path, subdirs, files in os.walk(DIR_IMAGES):
        imgNumber = 0
        for name in files:
            if name.lower().endswith(".jpg"):
                try:
                    filePath = path + "\\" + name
                    imageView = path.split("\\")[-1]
                    imageGroup = path.split("\\")[-2]

                    imageResized = cropAndResizeImg(filePath)

                    if GENERATE_IMAGES:
                        newImages = generateImages(imageResized)
                        for img in newImages:
                            cv2.imwrite(("training_images\\%s\\%s\\%s.jpg"%(imageGroup,imageView,imgNumber)),img)
                            imgNumber +=1
                    else:
                        cv2.imwrite(("training_images\\%s\\%s\\%s.jpg"%(imageGroup,imageView,imgNumber)),newImages)
                        imgNumber += 1
                except Exception as e:
                    print(("ERROR: IMAGEM %s: "%name),str(e))
            #end if
        #end for
    #end for 
#end def

def generateImages(image):
    img_Crop = image[BORDER-1:BORDER+SIZE-1, BORDER-1:BORDER+SIZE-1]
            
    # Generate all possible non-repeating pairs
    numbers = [i for i in range(0,51)]
    pairs = np.array(list(itertools.combinations(numbers, 2)))
    random.shuffle(pairs)
            
    newImages = []
            
    for i in range(0,N_NEW_IMAGES):
        [x,y] = random.choice(pairs)
        np.delete(pairs,[x,y])
                
        left = x if x <= 24 else 50 - x
        right = 50 - x if x <= 24 else x
        top = y if y <= 24 else 50 - y
        bottom = 50 - y if y <= 24 else y

        newImages.append(cv2.copyMakeBorder(img_Crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255,255,255])) 
        
    return newImages
    #end for
#end def  

def cropAndResizeImg(image):
    img_BGR = np.array(cv2.imread(image))

    mask = cv2.inRange(img_BGR, THRESHOLD_LOWER, THRESHOLD_UPPER)
    img_threshold = cv2.bitwise_and(img_BGR,img_BGR, mask= mask)

    img_Crop = cropImageObject(img_threshold,img_BGR)

    img_resized = cv2.resize(img_Crop, (SIZE,SIZE))

    img_Border10 = cv2.copyMakeBorder(img_resized, BORDER, BORDER, BORDER, BORDER, cv2.BORDER_CONSTANT, value=[255,255,255])
        
    return img_Border10

def cropImageObject(img_threshold, img):
    img_Gray = cv2.cvtColor(img_threshold, cv2.COLOR_BGR2GRAY)
    x,y,w,h = cv2.boundingRect(img_Gray)
            
    if h > w:
        if h % 2 == 0:
            return img[y:y+h, x-(h-w)//2:x+w+(h-w)//2]
        else:
            return img[y:y+h, x-(h-w)//2:x+w+(h-w)//2+1]
        #end if
    elif h < w:
        if h % 2 == 0:
            return img[y-(w-h)//2:y+h+(w-h)//2, x:x+w]
        else:
            return img[y-(w-h)//2:y+h+(w-h)//2+1, x:x+w]
        #end if
    elif h == 2:
        return img[y:y+h, x:x+w]
    #end if
#end def