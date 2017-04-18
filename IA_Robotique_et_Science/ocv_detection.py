#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging
import traceback
import cv2 
import datetime
import os

# la liste des classifiers d'OpenCV que nous utiliserons
TRAINSET_FACE_FRONTAL = "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"
TRAINSET_FACE_PROFILE = "/usr/share/opencv/haarcascades/haarcascade_profileface.xml"
TRAINSET_BODY_FULL = "/usr/share/opencv/haarcascades/haarcascade_fullbody.xml"

# des paramétres liés à la taille des images
DOWNSCALE = 1.05           # ratio appliqué à l'image. La valeur doit être > 1
MAX_SIZE = 800      # taille maximale de l’image en pixels


class OpenCVGenericDetection:

    def __init__(self, image_path, archive_folder = "/tmp/", debug = False):
        """ init
            @image_path : le chemin d'une image sur le disque
            @archive_folder : dossier d'archive
            @debug : si True, affichage des images dans une fenêtre
        """
        logging.info("Image : {0}".format(image_path))
        self.image_path = image_path
        self.archive_folder = archive_folder
        self.debug = debug
        self.items = []
        self.items_frames = []

        # On initialise le classifier
        self.set_classifier()

        # Afin de grouper les archives, nous allons utiliser un préfixe unique
        self.images_prefix = "{0}_{1}_".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"), self.__class__)

        # On charge l'image dans une frame 
        self.frame = cv2.imread(image_path)
        logging.info("Résolution de l'image : '{0}x{1}'".format(self.frame.shape[1], self.frame.shape[0]))

        # On vérifie si l'image est trop grande et si c'est le cas on calcule un ratio pour la réduire
        ratio = 1
        if self.frame.shape[1] > MAX_SIZE or self.frame.shape[0] > MAX_SIZE:
            if self.frame.shape[1] / MAX_SIZE > self.frame.shape[0] / MAX_SIZE:
                ratio = float(self.frame.shape[1]) / MAX_SIZE
            else:
                ratio = float(self.frame.shape[0]) / MAX_SIZE
     
        ## Si l'image est trop grande, on la retaille
        if ratio != 1:
            newsize = (int(self.frame.shape[1]/ratio), int(self.frame.shape[0]/ratio))
            logging.info("Redimensionnement de l'image en : {0}".format(newsize))
            self.frame = cv2.resize(self.frame, newsize) 

        # Affichage de l'image originale
        if self.debug:
            cv2.imshow("preview", self.frame) 
            cv2.waitKey()


    def set_classifier(self):
        """ Méthode à surcharger
        """
        self.classifier = None


    def find_items(self):
        """ Trouver les items dans une frame

            Valorise self.items en tant que liste contenant les coordonnées des viages au format (x, y, h, w).
            Exemple : 
                      [[ 483  137   47   47]
                       [ 357  152   46   46]
                       ...
                       [ 126  167   51   51]]

        """
        logging.info("Recherche des items...")

        # On applique le classifier pour détecter les visages
        items = self.classifier.detectMultiScale(self.frame, scaleFactor = DOWNSCALE, minNeighbors = 6, minSize=(40,40), maxSize=(150,150))
        
        # On valorise self.items et on affiche un peu de log
        logging.info("Nombre d'items : '{0}'".format(len(items)))
        logging.info("Items = {0}".format(items))
        self.items = items 


    def extract_items_frames(self):
        """ Extraire les frames des items de la frame complète
            Valorise self.items_frames en tant que liste des frames et coordonnées.
            Exemple : 
                      [
                        {  "frame" : ...,
                           "x" : ...,
                           "x" : ...,
                           "x" : ...,
                           "x" : ...
                        },
                        { ... },
                        ...
                      ]
        """
        logging.info("Extractions des frames des items ('{0}' à extraire)...".format(len(self.items)))
        items_frames = []
        # pour chaque coordonnées d'items...
        for f in self.items: 
            # On extrait le sous ensemble de la frame complète
            x, y, w, h = f 
            item_frame = self.frame[y:y+h,x:x+w]
            # Et on le stocke ainsi que ses coordonnées
            items_frames.append( {
                                   "frame" : item_frame,
                                   "x" : x,
                                   "y" : y,
                                   "w" : w,
                                   "h" : h,
                                 })

            # On affiche chaque visage extrait dans une fenêtre
            #if self.debug:
            #    cv2.imshow("preview", item_frame)
            #    cv2.waitKey()

        self.items_frames = items_frames


    def get_items_frames(self, grayscale = False):
        """ Retourne les frames des items et leurs coordonnées dans une liste
            @grayscale : si True, retourne les frames des items en niveaux de gris
        """
        # Si on ne désire pas un retour en niveau de gris, on retourne les données telles quelles
        if not grayscale:
            return self.items_frames

        # Dans le cas contraire, on créée une liste temporaire et pour chaque frame de la liste 
        # originale, on insère dans la nouvelle liste la frame convertie en niveaux de gris
        items_frames = []
        for item_frame in self.items_frames:
            item_frame["frame"] = cv2.cvtColor(item_frame["frame"], cv2.COLOR_BGR2GRAY)
            items_frames.append(item_frame)
        return items_frames

    def add_label(self, text, x, y):
        """ Ajout d'un label sur la frame complète
            @text : texte à afficher
            @x, y : coordonnées du texte à afficher
        """
        # Comme cette fonction sera utilisée pour afficher un label sur un carré autour d'un visage,
        # on ajoute volontairement un peu d'espace entre le label et le carré, sauf si on est au bord de l'image 
        if y > 11:
            y = y - 5
        cv2.putText(self.frame, text, (x, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2) 



    def archive_items_frames(self):
        """ Ecrit dans le dossier d'archive chaque frame de chaque item en tant qu'une image
        """
        logging.info("Archive les items ('{0}' à archiver)...".format(len(self.items_frames)))
        idx = 0
        # Pour chaque item, on le sauve dans un fichier 
        for item_frame in self.items_frames:
            a_frame = item_frame["frame"]
            image_name = "{0}_item_{1}.jpg".format(self.images_prefix, idx)
            logging.info("Archive un item dans le fichier : '{0}'".format(image_name))
            cv2.imwrite(os.path.join(self.archive_folder,  image_name), a_frame)
            idx += 1


    def archive_with_items(self):
        """ Ecrit dans le dossier d'archive la frame complète avec des carrés dessinés autour
            des visages détectés
        """
        logging.info("Archive l'image avec les items trouvés...")
        # Dessine un carré autour de chaque item
        for f in self.items: 
            x, y, w, h = f #[ v for v in f ] 
            cv2.rectangle(self.frame, (x,y), (x+w,y+h), (0,255,0), 3) 

        # Ajoute la date et l'heure à l'image
        cv2.putText(self.frame, datetime.datetime.now().strftime("%c"), (5, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3) 
            
        # On affiche l'image qui va être archivée dans une fenêtre
        if self.debug:
            cv2.imshow("preview", self.frame) 
            cv2.waitKey() 

        # Ecriture du fichier
        archive_full_name = "{0}_full.jpg".format(self.images_prefix)
        logging.info("Archive file is : '{0}'".format(archive_full_name))
        cv2.imwrite(os.path.join(self.archive_folder,  archive_full_name), self.frame)



class OpenCVFaceFrontalDetection(OpenCVGenericDetection):
    
    def set_classifier(self):
        """ To be overriden
        """
        self.classifier = cv2.CascadeClassifier(TRAINSET_FACE_FRONTAL)




if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.info("Test...") 
    T = OpenCVFaceFrontalDetection("./test/test.jpg",
                                   archive_folder = "./archives/",
                                   debug = False)
    T.find_items()
    T.extract_items_frames()
    T.archive_items_frames()
    T.archive_with_items()

