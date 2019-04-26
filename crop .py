# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from PIL import Image
from PIL.Image import *


from keras.layers import Dropout
import numpy as np


# Debut du decompte du temps
start_time = time.time()

#------------------------ Training CNN ---------------------------------

classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape = (50, 50, 3), activation = 'relu'))


classifier.add(MaxPooling2D(pool_size = (2, 2)))


classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
 
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())


classifier.add(Dense(units = 128, activation = 'relu'))

#cette partie -->
classifier.add(Dropout(0.2))

classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.2))

classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.2))
#cette partie <--

classifier.add(Dense(units = 1, activation = 'sigmoid'))


# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('datasez\\training_set',
                                                 target_size = (50, 50),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('datasez\\test_set',
                                            target_size = (50, 50),
                                            batch_size = 32,
                                            class_mode = 'binary')

history = classifier.fit_generator(training_set,
                         steps_per_epoch = 40,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 9 )


# plot metrics
pyplot.plot(history.history['acc'])
pyplot.show()

#-----------------------------------Fin training---------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
#----------------------------------Echantillonnage par Image --------------------------------------------------
from PIL import Image
from PIL.Image import *

import numpy as np
from keras.preprocessing import image
import time 

#tmps1=time.clock()
# Debut du decompte du temps
start_time = time.time()

img =open("rev.png")#chemin image  


r,g,b,vect= Image.getpixel(img, (1, 1))
print(r,g,b,vect)
    
r=0
t=0
for i in range (20):
    for j in range (25):
        
        e=0
        f=0
        coor_block=(r,t,r+100,t+100)
        
        
        memo = []
        cropped_block=img.crop(coor_block)


#save 10 imagette  de la diago car pas su comment faire la boucle qui change le nom du fichier a sauvegaerder         
        coor_diag=(e,f,e+50,f+50)
        cropped_imagette=cropped_block.crop(coor_diag)
        cropped_imagette.save("datasez\\single_prediction\\1.png")
        e+=50
        f+=50
        

        test_image = image.load_img('datasez\\single_prediction\\1.png', target_size = (50, 50))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = classifier.predict(test_image)
        training_set.class_indices
        
        if result[0][0] == 1:
            addit = 0
            prediction = 'saine'
        else:
            addit = 1
            prediction = 'malade'
        #print('prediction = ',prediction)
        
        #print("result = ", addit)
        
        memo.append(addit)
        
       
        
        coor_diag=(e,f,e+50,f+50)
        cropped_imagette=cropped_block.crop(coor_diag)
        cropped_imagette.save("datasez\\single_prediction\\2.png")
        e+=50
        f+=50


        test_image = image.load_img('datasez\\single_prediction\\2.png', target_size = (50, 50))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = classifier.predict(test_image)
        training_set.class_indices
        
        if result[0][0] == 1:
            addit = 0
            prediction = 'saine'
        else:
            addit = 1
            prediction = 'malade'
        #print('prediction = ',prediction)
        
        #print("result = ", addit)
        
        memo.append(addit)

        somme = sum(memo)
        
        #print('la somme des imgette malades dans le bloc ',r,t,' est ', somme)
        
        listcol=[]
        z=r
        listcol.append(z)
        for i in range (99):
            z=z+1
            listcol.append(z)
        
        
        listlig=[]
        w=t
        listlig.append(w)
        for i in range (99):
            w=w+1
            listlig.append(w) 
            
        
        if somme > 1:
            for y in listlig:
                for x in listcol:
                    rr,gg,bb,vv = Image.getpixel(img, (x, y))
                    Image.putpixel(img, (x, y), (255-rr,255-gg,255-bb))
        
        
        
        
        r+=100
    r=0
    t+=100

Image.show(img)
tmps2=time.clock()
#print (" temps excusion",tmps2-tmps1)

# Affichage du temps d execution
print("Temps d execution : %s secondes ---" % (time.time() - start_time))



#--------------------------------------------------------------------------------------------------------------





















