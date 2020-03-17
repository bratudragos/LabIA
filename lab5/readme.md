Descrierea algoritmului folosit:

Modelul bag-of-words:
este o metodă de reprezentare a datelor de tip text, bazată pe frecvența de apariție a cuvintelor în cadrul documentelor

Algoritmul este alcătuit din 2 pași:
1. definirea unui vocabular prin atribuirea unui id unic fiecărui
cuvânt regăsit în setul de date (setul de antrenare)
2. reprezentarea fiecărui document ca un vector de dimensiune egală cu lungimea vocabularului, definit astfel:
features[id_x] = numarul de aparitii al cuvantului cu id-ul idx

Normalizarea datelor
Standardizarea - transformă vectorii de caracteristici astfel încât fiecare să aibă medie 0 și deviație standard 1
Normalizarea L1. Normalizarea L2 - scalarea individuală a vectorilor de caracteristici corespunzători fiecărui exemplu astfel încât norma lor să devină 1

Mașini cu vectori suport
Pentru implementarea acestui algoritm vom folosi biblioteca ScikitLearn. Aceasta este dezvoltată în Python, fiind integrată cu NumPy și pune la dispoziție o serie de algoritmi optimizați pentru probleme de clasificare, regresie și clusterizare

Functii utilizate in program:
normalize_data(train_data, test_data, type=None) care primește ca parametri datele de antrenare, respectiv de testare și tipul de normalizare și întoarce aceste date normalizate

topCoef(classifier,feature_names,top_features=10) care primeste ca parametrii classifier-ul,cuvintele si primele cate cuvinte extragem, functia ordoneaza si afiseaza primele top_features cuvinte spam si primele top_features cuvinte non-spam

Clasa BagOfWords si metode ale acestei clase:

Are drept campuri vocabularul de cuvinte retinute, lista cu cuvintele in sine asezate in ordinea aparitiei in document si lungimea vocabularului

build_vocabulary(self, data) care primește ca parametru o listă de mesaje(listă de liste de strings) și construiește vocabularul pe baza acesteia

get_features(self, data) care primește ca parametru o listă de mesaje de dimensiune num_samples(listă de liste de strings) și returnează o matrice de dimensiune (num_samples x dictionary_length) definită astfel:
features(word_idx,sample_idx) =numarul de aparitii al cuvantului cu id word_idx in documentul sample_idx



