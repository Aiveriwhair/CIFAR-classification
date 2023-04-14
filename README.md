# CIFAR-classification

- V1 : Instrumentation et évaluation "en continu" du système
- V2 : Augmenter le nombre de cartes (filtres) dans la première couche de convolution de 6 à 12. Ajouter une couche de convolution supplémentaire avec 16 cartes, suivie d'une couche complètement connectée avec 128 neurones. Entraîner le modèle pendant 2 itérations.
- v3 : Ajouter une couche de convolution supplémentaire avec 32 cartes et une taille de filtre de 3x3, suivie d'une couche complètement connectée avec 256 neurones. Entraîner le modèle pendant 2 itérations.
- V4 : Remplacer la deuxième couche de convolution par une couche de convolution avec 32 cartes et une taille de filtre de 3x3. Ajouter une couche complètement connectée avec 512 neurones. Entraîner le modèle pendant 2 itérations.
- V5 : Ajouter une couche de convolution supplémentaire avec 64 cartes et une taille de filtre de 3x3, suivie d'une couche de max pooling avec une fenêtre de 2x2. Ajouter une couche complètement connectée avec 512 neurones. Entraîner le modèle pendant 2 itérations.
