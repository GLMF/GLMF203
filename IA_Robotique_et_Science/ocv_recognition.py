#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging
import traceback
import cv2 
import datetime
import os
import sys
import numpy


class OpenCVGenericRecognition:

    def __init__(self, trainset_path, archive_folder = "/tmp/"):
        """ Constructeur
            @trainset_path : le chemin vers le jeu de données
            @archive_folder : le dossier d'archive
        """
        logging.info("Trainset : {0}".format(trainset_path))
        self.trainset_path = trainset_path
        self.archive_folder = archive_folder

        # Taille de conversion des images du jeu de données
        # Ceci permettra de mettre des images de tailles variables dans le jeu de données
        self.resize_faces = (170, 170)   # valeur prise un peu au hasard je dois l'avouer

        # Initialisations diverses
        self.model = None
        self.trainset_images = []
        self.trainset_index = []
        self.trainset_identities = []


    def load_trainset(self):
        """ Chargement en mémoire du jeu de données
        """
        logging.info("Chargement du trainset...")

        c = 0 
        self.trainset_images = []       # images chargées
        self.trainset_index = []        # index (numéro) de l'identité
        self.trainset_identities = []   # identités du jeu de données

        # On parcourt le dossier contenant le jeu de données.
        # Il doit contenir des dossiers. Le nom du dossier sera le nom de la perosnne, son ientité.
        # Chaque dossier contiendra des images du visage de la personne.
        for dirname, dirnames, filenames in os.walk(self.trainset_path): 
            for subdirname in dirnames: 
                self.trainset_identities.append(subdirname) 
                logging.info("- identité '{0}'...".format(subdirname))
                subject_path = os.path.join(dirname, subdirname) 
                for filename in os.listdir(subject_path): 
                    try: 
                        # On convertit en niveau de gris et on redimensionne dans une dimension commune
                        im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE) 
                        im = cv2.resize(im, self.resize_faces) 
                        # On convertit l'image en 'array'
                        self.trainset_images.append(numpy.asarray(im, dtype=numpy.uint8)) 
                        # Et on dit que cet array correspond à l'identité n°c
                        self.trainset_index.append(c) 
                    except IOError, (errno, strerror): 
                        logging.error("I/O error({0}): {1}".format(errno, strerror))
                    except: 
                        logging.error("Unexpected error:", sys.exc_info()[0]) 
                        raise 
                c = c+1 


    def train(self): 
        """ Entraînement du jeu de données
            Méthode à surcharger
        """
        self.model = None


    def add_to_trainset(self):
        # TODO
        pass


    def recognize(self, frame):
        """ Applique la reconnaissance à une frame d'un visage
            @param frame : la frame
        """
        # Redimensionnement de la frame du visage aux dimensions du trainset
        frame = cv2.resize(frame, self.resize_faces) 

        # Reconnaissance
        # confidence = 0 ==> score parfait ! Plus la confidence est faible, plus on est sûr de l'identitié
        [idx, confidence] = self.model.predict(frame) 
        print(confidence)
        found_identity = self.trainset_identities[idx] 
        if confidence < 120:
            identity = found_identity
            found = True
        else:
            identity = "n/a ({0})".format(found_identity)
            found = False
        return found, identity, int(confidence)
        

 



        


class OpenCVFaceRecognitionLBPH(OpenCVGenericRecognition):
    
    def train(self): 
        """ Entraînement du jeu de données
            Méthode à surcharger
        """
        logging.info("Entraînement du trainset...")
        #self.model = cv2.createFisherFaceRecognizer() 
        #self.model = cv2.createEigenFaceRecognizer()
        self.model = cv2.createLBPHFaceRecognizer() 
        #self.model = cv2.createLBPHFaceRecognizer(radius = 1, grid_x = 6, grid_y = 6)
        self.model.train(numpy.asarray(self.trainset_images), numpy.asarray(self.trainset_index)) 






if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.info("Test...") 

    from ocv_detection import OpenCVFaceFrontalDetection
    R = OpenCVFaceRecognitionLBPH("./samples/faces/",
                              archive_folder = "./archives/")
    R.load_trainset()
    R.train()

    # test on some faces

    dir = "./test/"
    for filename in os.listdir(dir):
        filepath = os.path.join(dir, filename)
        T = OpenCVFaceFrontalDetection(filepath,
                                       archive_folder = "./archives/",
                                       debug = False)
        T.find_items()
        T.extract_items_frames()
        for item in T.get_items_frames(grayscale = True): 
            known, identity, confidence = R.recognize(item["frame"])
            label = "{0} ({1})".format(identity, confidence)
            logging.info("Trouvé : {0}".format(label))
            x = item["x"] 
            y = item["y"] 
            T.add_label(label, x, y)
            if not known:
                pass
                # TODO : save in a unknown folder
                # TODO : save in a unknown folder
                # TODO : save in a unknown folder
        T.archive_items_frames()
        T.archive_with_items()

