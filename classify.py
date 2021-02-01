import pandas as pd 
import numpy  as np
import copy
from sklearn.datasets import load_svmlight_file
#metricas
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#import classificadores
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.lda import LDA

def main():
	#carregando dados
	X_train, y_train = load_svmlight_file('./Archive/train.txt') 
	X_test,  y_test  = load_svmlight_file('./Archive/test.txt')

	X_train = X_train.toarray()
	X_test  = X_test.toarray()

	#lista para concatenar train_size e acuracias
	accs = []
	#knn classifier
	knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
	bayes = GaussianNB() 		   #naive bayes classfier
	lda_clf = LinearDiscriminantAnalysis() 	#linear discriminant analyses - LDA
	log_clf = LogisticRegression() #Logistic Regression


	for train_size in range(1000, len(X_train) + 1000, 1000):
		
		Xx_train = X_train[:train_size]
		yy_train = y_train[:train_size]

		print("Treinando KNN para entrada: %i") %(train_size) 
		#knn classifier
		knn.fit(Xx_train, yy_train)
		knn_pred = knn.predict(X_test)
		knn_acc  = accuracy_score(y_test, knn_pred)
		print 'Acuracia: %f' %(knn_acc)
		print confusion_matrix(y_test, knn_pred)
		print ' '

		print("Treinando Naive Bayes para entrada: %i") %(train_size) 
		#Naive Bayes classifier
		bayes_pred = bayes.fit(Xx_train, yy_train).predict(X_test)
		bayes_acc = accuracy_score(y_test, bayes_pred)
		print 'Acuracia: %f' %(bayes_acc)
		print confusion_matrix(y_test, bayes_pred)
		print ' '

		print("Treinando LDA para entrada: %i") %(train_size) 
		#linear discrimant analyses - LDA
		lda_pred = lda_clf.fit(Xx_train, yy_train).predict(X_test)
		lda_acc  = accuracy_score(y_test, lda_pred)
		print 'Acuracia: %f' %(lda_acc)  
		print confusion_matrix(y_test, lda_pred)
		print ' '

		print("Treinando Logistic Regression para entrada: %i") %(train_size) 
		#Logistic Regression
		log_pred = log_clf.fit(Xx_train, yy_train).predict(X_test)
		log_acc  = accuracy_score(y_test, log_pred)
		print 'Acuracia: %f' %(log_acc) 
		print confusion_matrix(y_test, log_pred)
		print ' '

		accs.append(copy.deepcopy([copy.deepcopy(train_size), copy.deepcopy(knn_acc), copy.deepcopy(bayes_acc), copy.deepcopy(lda_acc), copy.deepcopy(log_acc)]))

	accs = np.array(accs)
	df = pd.DataFrame({'train_size': accs[:,0], 'knn_acc': accs[:, 1], 'bayes_acc': accs[:,2], 'lda_acc': accs[:,3], 'log_acc': accs[:,4]})
	df.to_csv("acc_classifiers.csv",index= False)

if __name__ == "__main__":
        # if len(sys.argv) < 2:
        #         sys.exit("Use: knn.py <data>")

        main()
