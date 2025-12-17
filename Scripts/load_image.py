import os
import numpy as np
from PIL import Image

#dimensione uniforme delle immagini
def load_images_from_folders(root_dir, target_size=(300,300)):
    #conterrà tutte le immagini come array numerici
    images = []
    #indici numerici delle classi, sarà usato come target dall'algoritmo
    labels = []
    #nome cartella che indicano le classi, ignora le cartelle nascoste
    #sorted garantisce l'ordine deterministico
    class_names = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith(".")
    ])

    for label, class_name in enumerate(class_names):
        #costruisce il path delle classi
        class_path = os.path.join(root_dir, class_name)

        #esclude i path non appartenenti alle classi
        if not os.path.isdir(class_path):
            continue

        for file in os.listdir(class_path):
            #prende solo i file immagini
            if file.lower().endswith(('.jpg', '.png')):
                #path completa dell'immagine
                img_path = os.path.join(class_path, file)
                
                #conversione in scala RGB, da cambiare se non si lavora su RGB
                img = Image.open(img_path).convert("RGB")
                #adattamento immagini al size per necessità
                img = img.resize(target_size)
                #conversione in array con normalizzazione dei pixel
                img = np.array(img) / 255.0

                #salva le informazioni negli array
                images.append(img)
                labels.append(label)

    return np.array(images), np.array(labels), class_names

'''
#TEST FUNZIONE
X, y, class_names = load_images_from_folders("Rock-Paper-Scissors/test/")

print("Numero totale immagini:", X.shape[0])
#restituisce il totale una tupla di tipo: (numero immagini, dimensione immagini (NxM), canali colori RGB)
print("Shape immagini:", X.shape) 
print("Shape labels:", y.shape)       
print("Classi:", class_names)
print("Numero classi:", len(class_names))
'''
