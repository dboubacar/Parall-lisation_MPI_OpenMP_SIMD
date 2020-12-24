Le but du projet est de paralléliser un programme séquentiel qui effectue la résolution 
de systèmes linéaires du type Ax = b,où A est une matrice creuse réellesymétrique définie positive. 
Dans toute la suite, n désigne la taille du système.

Ce programme utilise la méthode du gradient conjugué. Il s’agit d’une (ancienne) méthode itéra
tive : le calcul se fait principalement en calculant des produits matrice-vecteur avec la matrice
A. L’avantage de cette famille d’algorithmes, c’est qu’une fois que la matrice A tient en mémoire
alors on peut lancer le calcul et la consommation mémoire ne va plus augmenter. A contrario,
elle peut exploser dans les algorithmes d’élimination gaussienne creuse.

Ceci permet de résoudre de très grands systèmes linéaires creux, par exemple avec n de l’ordre
de plusieurs millions, sur des ordinateurs personnels. Si ces systèmes étaient denses, il faudrait
des centaines de Tera-octets ne serait-ce que pour stocker la matrice A !
