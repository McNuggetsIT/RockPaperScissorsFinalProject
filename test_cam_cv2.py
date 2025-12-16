import cv2 as cv

#avvia la telecamera
#0 indica la fotocamera predenfinita
cap = cv.VideoCapture(0)

#errore se non riesce ad aprire la telecamera
if not cap.isOpened():
    print("Errore apertura camera")
    exit()

while True:
    #ret indica se il frame è stato letto correttamente
    #frame sarà il frame che conterrà l'immagine
    ret, frame = cap.read()
    if not ret:
        break

    #rimuove effetto specchio con 1
    frame = cv.flip(frame, 1)

    #mostra la fotocamera
    cv.imshow("Webcam", frame)

    #aspetta il tasto q per uscire
    #0xFF serve per compatibilità di sistema operativo
    #27 = ESC
    key = cv.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

#rilascia la webcam e chiude le finestre create
cap.release()
cv.destroyAllWindows()