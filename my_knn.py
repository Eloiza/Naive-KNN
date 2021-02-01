
#calcula distancia euclidiana entre dois vetores
def euclidean_distance(v1, v2):
	total = 0
	for i in range(len(v1)):
		middle_sum = (v1[i] - v2[i])
		total += middle_sum**2 
	
	return total**(0.5)


#encontra as k amostras mais proximas da amostra X_test
def find_neighbour(X_train, y_train, sample, k):
	k_nearestn = [None for x in range(k)] #guarda os k vizinhos mais prox

	#calculo todas as distancias de sample para o dataset de treino
	for i in range(len(X_train)):
		distance = euclidean_distance(X_train[i], sample) #calculo a distancia de sample para outra amostra
		for n in range(k): #verifico se nova distancia e mais perto que as k anteriores
			if(k_nearestn[n] == None or distance < k_nearestn[n][0]):
				k_nearestn[n] = (distance, y_train[i])

	ret = []
	for i in range(0,k):
	 	ret.append(int(k_nearestn[i][1]))
	
	return ret #retorno classe das menores distancias

#encontra o rotulo baseado em um calculo de histograma
def find_label(neighbours, labels):
	hist = [[0,i] for i in labels] #vetor para guardar (total_de_ocorrencias, label)
	
	for i in range(len(neighbours)):
		for lb in range(len(labels)):
			if(int(neighbours[i]) == lb):
				hist[lb][0] += 1

	hist.sort(reverse=True)
	return hist[0][1] #retorno rotulo de maior ocorrencia

#calcula acuracia 
def acc_score(pred, true):
	correct_preds = 0
	#oq acertou pelo tudo oq fez
	for i in range(len(pred)):
		if(int(pred[i]) == true[i]):
			correct_preds +=1

	return (float(correct_preds)/len(pred))

def confusion_matrix(pred, true, label):
	n_labels = len(label)
	#criando matrix n_labels**2 preenchida com 0
	matrix = [[0 for x in range(n_labels)] for x in range(n_labels)]
	
	for i in range(len(pred)):
		x = int(pred[i])
		y = int(true[i])
		matrix[x][y] += 1

	return matrix

