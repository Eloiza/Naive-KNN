from sklearn.datasets import load_svmlight_file
from sklearn.utils.multiclass import unique_labels
import argparse
import sys 

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

def main(train_file, test_file, k):

	#carregando base de dados
	try:
		X_train, y_train = load_svmlight_file(train_file)
	except:
		print "File %s not found" %(train_file)
		sys.exit(0)

	try:
		X_test, y_test = load_svmlight_file(test_file)
	except:
		print "File %s not found" %(test_file)
		sys.exit(0)

	#convertendo matriz esparsa para numpy array
	X_test = X_test.toarray()
	X_train = X_train.toarray()

	print 'Dados Carregados \nIniciando Treinamento'

	#inicializacoes
	y_pred = []
	label = 0 
	unique_lb = list(unique_labels(y_train))

	for i in range(len(X_train)):
		neighbours = find_neighbour(X_train, y_train, X_test[i], k)  #retorna a classe dos k vizinhos mais proximos
		label = find_label(neighbours, unique_lb)	#decide a classe baseado noso rotulo dos k vizinhos
	 	y_pred.append(label)

	print 'Classificando Amostras...'
	accuracy = acc_score(y_pred, y_test)
	cm = confusion_matrix(y_pred, y_test, unique_lb)

	print 'Acuracia: %f' %(accuracy)
	print 'Confusion Matrix'
	for x in cm:
		print x


if __name__ == "__main__":
	#tratamento de parametros de entrada pela linha de comando
	parser = argparse.ArgumentParser(description='k-nearest neighbour algorithm')
	parser.add_argument('train_file', type=str, help = 'train dataset file name')
	parser.add_argument('test_file',  type=str, help = 'test dataset file name')
	parser.add_argument('k', type=int, help = 'k neighbours to the algorithm')
	args = parser.parse_args()

	print args
	main(args.train_file, args.test_file, args.k)
