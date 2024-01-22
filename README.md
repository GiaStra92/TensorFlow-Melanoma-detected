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


**Creazione del modello:**

È cruciale notare che è fondamentale fornire al modello immagini con dimensioni uniformi. La scelta di queste dimensioni è arbitraria, e per questo specifico modello abbiamo optato per immagini ridimensionate a 150x150 pixel, rendendole così tutte quadrate.

Considerando che lavoriamo con immagini a colori, è essenziale incorporare questa informazione nella nostra progettazione. Pertanto, la forma di input sarà (150, 150, 3), dove il valore 3 rappresenta i tre canali di colore. Successivamente, vedremo come garantire che tutte le nostre immagini siano conformi a questa dimensione quando sfruttiamo ImageDataGenerator.

Ora procediamo con l'implementazione dell'architettura della rete neurale.
Nota bene: il seguente codice viene eseguito su Google Colab dato che richiede molte risorse computazionali.

installiamo la libreria opendataset

<img width="1440" alt="image" src="https://github.com/GiaStra92/TensorFlow-Melanoma-detected/assets/140896994/b1e097e4-71b5-48b2-9f99-553db1b07014">

 
 scarichiamo il dataset vi ricordo che avrete bisogno della api-key di kaggle
 
<img width="1376" alt="Screenshot 2024-01-19 alle 15 45 18" src="https://github.com/GiaStra92/TensorFlow-Melanoma-detected/assets/140896994/65972a27-f2d1-4485-aa04-7a2a9f71d7da">

```python

import os
# importante rinominare path melanoma/DermMel/train_sep in :
# /content/melanoma/DermMel/train




import os

current_path = '/content/melanoma/DermMel/train_sep'
new_path = '/content/melanoma/DermMel/train'

if not os.path.exists(new_path):
   
    os.rename(current_path, new_path)
    print(f"La cartella è stata rinominata da {current_path} a {new_path}")
else:
    print(f"Il percorso di destinazione {new_path} esiste già. Non è necessario rinominare la cartella.")





# cartelle di apprendimento e validazione

base_dir = '/content/melanoma/DermMel'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'valid')

# puntiamo alle cartelle di Melanoma e NotMelanoma per il training
train_Melanoma_dir = os.path.join(train_dir, 'Melanoma')
train_NotMelanoma_dir = os.path.join(train_dir, 'NotMelanoma')

# puntiamo alle cartelle di Melanoma e NotMelanoma per la validazione
validation_Melanoma_dir = os.path.join(validation_dir, 'Melanoma')
validation_NotMelanoma_dir = os.path.join(validation_dir, 'NotMelanoma')





import tensorflow as tf

# Creiamo un modello sequenziale
model = tf.keras.models.Sequential([
    # Primo strato di convoluzione con 16 filtri 3x3, attivazione ReLU e dimensione dell'input 150x150x3
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),

    # Strato di MaxPooling 2x2
    tf.keras.layers.MaxPooling2D(2,2),

    # Secondo strato di convoluzione con 32 filtri 3x3 e attivazione ReLU
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

    # Altro strato di MaxPooling 2x2
    tf.keras.layers.MaxPooling2D(2,2),

    # Terzo strato di convoluzione con 64 filtri 3x3 e attivazione ReLU
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    # Ulteriore strato di MaxPooling 2x2
    tf.keras.layers.MaxPooling2D(2,2),

    # Appiattiamo il risultato per passarlo ad uno strato denso
    tf.keras.layers.Flatten(),

    # Strato denso con 512 neuroni e attivazione ReLU
    tf.keras.layers.Dense(512, activation='relu'),

    # Strato di output con un singolo neurone e attivazione sigmoide
    tf.keras.layers.Dense(1, activation='sigmoid')
    #output 0 benigno 1 maligno
])

# Visualizziamo una panoramica del modello
model.summary()



```

**output**

<img width="1437" alt="image" src="https://github.com/GiaStra92/TensorFlow-Melanoma-detected/assets/140896994/633540a0-28f0-4492-9cd1-d97a6d365485">

Output Shape" in relazione al primo strato convoluzionale. Quando osserviamo che l'output shape è 148x148x64, dobbiamo considerare diversi aspetti.

Innanzitutto, l'immagine in input è di dimensioni 150x150. La riduzione di 1 pixel su ciascun lato dell'immagine durante la convoluzione è dovuta all'utilizzo di un filtro (kernel) di dimensione 3x3. Questo filtro scorre sull'immagine con passi (stride) di default, solitamente 1 pixel alla volta, riducendo così la dimensione dell'output.

Per capire meglio, consideriamo il centro del filtro 3x3. Se immaginiamo questo filtro posizionato sull'immagine, noteremo che il suo centro coincide con un punto sull'immagine. Tuttavia, nei bordi dell'immagine, non c'è spazio sufficiente per posizionare completamente il filtro senza superare i bordi. Quindi, per mantenere la dimensione dell'output compatibile con l'input, perdiamo un pixel su ciascun lato.

Quanto al numero 64 nell'output shape, rappresenta il numero di filtri (o kernel) applicati. Ogni filtro estrae differenti caratteristiche dall'immagine. Quindi, in questo caso, abbiamo 64 filtri convoluzionali applicati all'immagine di input, ognuno dei quali produce un'immagine di output 148x148.

In breve, l'output shape 148x148x64 nel primo strato convoluzionale indica che stiamo ottenendo 64 diverse immagini di output (o feature maps), ciascuna di dimensioni 148x148, applicando filtri 3x3 all'immagine di input di dimensioni 150x150. Questo processo di convoluzione aiuta la rete a identificare e catturare diverse caratteristiche dell'immagine durante le fasi iniziali del processo di apprendimento.
