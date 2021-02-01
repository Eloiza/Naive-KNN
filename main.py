import my_knn
import argparse
import sys 
from sklearn.datasets import load_svmlight_file
from sklearn.utils.multiclass import unique_labels

def main(train_file, test_file, k):
	#carregando base de dados
	try:
		X_train, y_train = load_svmlight_file(train_file)
	except:
		print("File %s not found" %(train_file))
		sys.exit(0)

	try:
		X_test, y_test = load_svmlight_file(test_file)
	except:
		print("File %s not found" %(test_file))
		sys.exit(0)

	#convertendo matriz esparsa para numpy array
	X_test = X_test.toarray()
	X_train = X_train.toarray()

	print('Dados Carregados \nIniciando Treinamento')

	#inicializacoes
	y_pred = []
	label = 0 
	unique_lb = list(unique_labels(y_train))

	for i in range(len(X_train)):
		neighbours = my_knn.find_neighbour(X_train, y_train, X_test[i], k)  #retorna a classe dos k vizinhos mais proximos
		label = my_knn.find_label(neighbours, unique_lb)	#decide a classe baseado noso rotulo dos k vizinhos
		y_pred.append(label)

	print('Classificando Amostras...')
	accuracy = my_knn.acc_score(y_pred, y_test)
	cm = my_knn.confusion_matrix(y_pred, y_test, unique_lb)

	print('Acuracia: %f' %(accuracy))
	print('Confusion Matrix')
	for x in cm:
		print(x)


if __name__ == "__main__":
	#tratamento de parametros de entrada pela linha de comando
	parser = argparse.ArgumentParser(description='k-nearest neighbour algorithm')
	parser.add_argument('train_file', type=str, help = 'train dataset file name')
	parser.add_argument('test_file',  type=str, help = 'test dataset file name')
	parser.add_argument('k', type=int, help = 'k neighbours to the algorithm')
	args = parser.parse_args()

	print(args)
	main(str(args.train_file), str(args.test_file), int(args.k))
