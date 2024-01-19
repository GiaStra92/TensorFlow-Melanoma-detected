# Classificazione di Melanoma in TensorFlow: Distinguere tra Maligno e Benigno con Intelligenza Artificiale

![](https://upload.wikimedia.org/wikipedia/commons/2/2d/Tensorflow_logo.svg)


TensorFlow è una libreria software open-source per l'apprendimento automatico, sviluppata da Google Brain Team. È utilizzata per implementare algoritmi di reti neurali, sia per scopi di ricerca che per applicazioni pratiche. TensorFlow offre un'ampia gamma di strumenti e risorse che permettono di costruire e addestrare modelli di machine learning complessi. In questo articolo esploreremo il processo di creazione di un modello di classificazione binaria utilizzando TensorFlow. L'obiettivo sarà distinguere melanoma benigno e melanoma maligno nelle immagini, prendendo ispirazione da un noto dataset su Kaggle. Questa esperienza ci fornirà l'opportunità di apprendere diversi aspetti importanti:

1. **Scaricare e decomprimere un dataset dal web.**
2. **Assemblare un modello di classificazione incorporando livelli di convoluzione e max pooling.**
3. **Implementare un ImageDataGenerator per ottimizzare la gestione delle immagini durante le fasi di training e validazione. Compilare e allenare il modello.**
5. **Esaminare le modifiche effettuate alle immagini attraverso i vari strati della rete neurale.**
6. **Effettuare previsioni su immagini inedite.**

Affronteremo queste sfide tenendo presente che il deep learning richiede risorse computazionali significative. Per questo, sfrutteremo Google Colab, impostando il runtime su GPU, per gestire efficientemente i calcoli necessari

Il set di dati è composto da 8.903 immagini di melanoma 8.902 immagini di non melanoma.
Url dataset https://www.kaggle.com/datasets/drscarlat/melanoma il seguente dataset fa riferimento ha seguente link [link](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T), vi ricordo che avete bisogno di API-key , per ottenere la seguente chiave basta registrarsi gratuitamente su https://www.kaggle.com. 

**Breve Cenni convoluzioni e max pooling** 

Le convoluzioni e il max pooling sono tecniche di elaborazione delle immagini fondamentali per i modelli di computer vision, specialmente nelle reti neurali convoluzionali (CNNs). Questi strumenti aiutano il modello a catturare e enfatizzare le caratteristiche salienti delle immagini, facilitando l'apprendimento di pattern complessi.

**Convoluzione:** Questo processo coinvolge l'applicazione di un filtro, o kernel, che passa attraverso l'immagine. Immaginate ogni pixel di un'immagine come una cellula in un vasto tessuto di informazioni visive. Un filtro convoluzionale esamina una cellula (pixel) insieme alle sue vicine immediate, calcolando un nuovo valore per la cellula centrale. Questo si ottiene attraverso una griglia, tipicamente di dimensione 3x3, che pesa il valore di ogni pixel e i suoi vicini secondo i valori specificati nel filtro. Il risultato è una trasformazione del pixel centrale che riflette non solo la sua intensità originale ma anche la relazione spaziale con i suoi vicini. Questo processo di filtraggio evidenzia i pattern locali, come bordi o angoli, che sono essenziali per la comprensione visiva.

_fonte immagine:_
https://www.developersmaggioli.it:
 ![gif](https://github.com/GiaStra92/TensorFlow-Melanoma-detected/assets/140896994/8a051c3f-3d47-4fb7-965b-1eb73803f68f)

La procedura di convoluzione in elaborazione delle immagini si basa sull'uso di un particolare array, noto come kernel o filtro. Questo kernel, solitamente più piccolo dell'immagine da elaborare, viene fatto scorrere su tutta l'immagine. Un esempio tipico è un kernel di dimensione 3x3 che si muove su un'immagine 5x5. Durante questo processo, ogni elemento dell'immagine viene combinato con il kernel, producendo una nuova immagine, chiamata mappa delle caratteristiche (o feature map), che qui è illustrata come una matrice 5x5 di colore verde.

Un dettaglio importante in questo processo è il cosiddetto padding, che è l'aggiunta di bordi extra (qui mostrati in bianco) attorno all'immagine originale. Il padding è usato per mantenere le dimensioni desiderate nell'immagine di output (nella nostra esemplificazione, una matrice 5x5). Senza il padding, l'immagine risultante sarebbe ridotta nelle sue dimensioni (nel nostro caso, a 3x3). Il padding non è solo una soluzione tecnica per conservare le dimensioni, ma aiuta anche a preservare le informazioni ai bordi dell'immagine, che altrimenti potrebbero essere perse nei livelli successivi di elaborazione. 


La concezione fondamentale di una convoluzione si focalizza sull'accentuare specifici attributi di un'immagine, per esempio esaltando i bordi e i contorni per renderli più distinti rispetto allo sfondo ecco un esempio.

 
![image](https://github.com/GiaStra92/TensorFlow-Melanoma-detected/assets/140896994/1eb605af-94d0-409a-a6a1-4ce818bf90c6)
_fonte immagine:_ https://www.developersmaggioli.it
 
**Max Pooling:** Dopo la convoluzione, spesso segue il pooling, che riduce le dimensioni dell'immagine mantenendo le caratteristiche più prominenti. Il max pooling, in particolare, scorre un filtro attraverso l'immagine e seleziona il valore massimo all'interno di una finestra di valori (ad esempio, all'interno di una griglia 2x2). Questo non solo riduce la complessità computazionale per le operazioni successive ma conserva anche le caratteristiche più rilevanti che sono state evidenziate dalla convoluzione.

Queste operazioni sono essenziali per permettere alle reti neurali di apprendere gerarchie di caratteristiche visive, da semplici texture e forme a oggetti complessi all'interno di immagini.

![image](https://github.com/GiaStra92/TensorFlow-Melanoma-detected/assets/140896994/01ef33b5-01c0-49c1-aa72-2e55ebcb64c6)


**Creazione del modello **

È cruciale notare che è fondamentale fornire al modello immagini con dimensioni uniformi. La scelta di queste dimensioni è arbitraria, e per questo specifico modello abbiamo optato per immagini ridimensionate a 150x150 pixel, rendendole così tutte quadrate.

Considerando che lavoriamo con immagini a colori, è essenziale incorporare questa informazione nella nostra progettazione. Pertanto, la forma di input sarà (150, 150, 3), dove il valore 3 rappresenta i tre canali di colore. Successivamente, vedremo come garantire che tutte le nostre immagini siano conformi a questa dimensione quando sfruttiamo ImageDataGenerator.

Ora procediamo con l'implementazione dell'architettura della rete neurale.
Nota bene: il seguente codice viene eseguito su Google Colab dato che richiede molte risorse computazionali.

installiamo la libreria opendataset


