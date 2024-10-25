import numpy as np  # for basic operations over arrays
from scipy.spatial import distance  # to compute the Gaussian kernel
import cvxopt  # to solve the dual optimization problem
import copy  # to copy numpy arrays
import torch


def linear_kernel(x, xʹ, c=0):
    return x @ xʹ.T

class SVM:
    kernel_funs = {'linear': linear_kernel}
    def __init__(self, kernel='linear', C=5, k=2):
        # set the hyperparameters
        self.kernel_str = kernel
        self.kernel = SVM.kernel_funs[kernel]
        self.C = C  # regularization parameter
        self.k = k  # kernel parameter

        # training data and support vectors
        self.X, y = None, None
        self.αs = None

        # for multi-class classification
        self.multiclass = False
        self.clfs = []

    def fit(self, X, y, eval_train=False, epochs=100):

        # Check if X and y are PyTorch tensors and convert them to NumPy arrays if necessary
        if torch.is_tensor(X):
            X = X.cpu().numpy()  # Moves to CPU and converts to NumPy, works for both CUDA and non-CUDA tensors

        if torch.is_tensor(y):
            y = y.cpu().numpy()  # Same as above

        # Now X and y are guaranteed to be NumPy arrays, so you can proceed with your existing logic
        if len(np.unique(y)) > 2:
            self.multiclass = True
            return self.multi_fit(X, y, eval_train)

        # relabel if needed
        if set(np.unique(y)) == {0, 1}:
            y[y == 0] = -1

        # ensure y has dimensions Nx1
        self.y = y.reshape(-1, 1).astype(np.double)  # Has to be a column vector
        self.X = X
        N = X.shape[0]

        # compute the kernel over all possible pairs of (x, x') in the data
        self.K = self.kernel(X, X, self.k)

        # For 1/2 x^T P x + q^T x
        P = cvxopt.matrix(self.y @ self.y.T * self.K)
        q = cvxopt.matrix(-np.ones((N, 1)))

        # For Ax = b
        A = cvxopt.matrix(self.y.T)
        b = cvxopt.matrix(np.zeros(1))

        # For Gx <= h
        G = cvxopt.matrix(np.vstack((-np.identity(N),
                                     np.identity(N))))
        h = cvxopt.matrix(np.vstack((np.zeros((N, 1)),
                                     np.ones((N, 1)) * self.C)))

        # Solve
        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        self.αs = np.array(sol["x"])

        # Maps into support vectors
        self.is_sv = ((self.αs > 1e-3) & (self.αs <= self.C)).squeeze()
        self.margin_sv = np.argmax((1e-3 < self.αs) & (self.αs < self.C - 1e-3))

        # Evaluate training accuracy at the end of training (you can move this inside a loop for periodic updates)
        # if eval_train:
        #     training_accuracy = self.evaluate(X, y) * 100
        #     print(f"Finished training with accuracy {training_accuracy:.2f}%")

    def multi_fit(self, X, y, eval_train=False):
        self.k = len(np.unique(y))  # number of classes
        # for each pair of classes
        for i in range(self.k):
            # get the data for the pair
            Xs, Ys = X, copy.copy(y)
            # change the labels to -1 and 1
            Ys[Ys != i], Ys[Ys == i] = -1, +1
            # fit the classifier
            clf = SVM(kernel=self.kernel_str, C=self.C, k=self.k)
            clf.fit(Xs, Ys)
            # save the classifier
            self.clfs.append(clf)
        # if eval_train:
        #     print(f"Finished training with accuracy {self.evaluate(X, y) * 100:.2f}%")

    def multi_fit(self, X, y, eval_train=False):
        self.k = len(np.unique(y))  # number of classes
        # for each pair of classes
        for i in range(self.k):
            # get the data for the pair
            Xs, Ys = X, copy.copy(y)
            # change the labels to -1 and 1
            Ys[Ys != i], Ys[Ys == i] = -1, +1
            # fit the classifier
            clf = SVM(kernel=self.kernel_str, C=self.C, k=self.k)
            clf.fit(Xs, Ys)
            # save the classifier
            self.clfs.append(clf)
        # if eval_train:
        #     print(f"Finished training with accuracy {self.evaluate(X, y) * 100:.2f}%")

    def multi_predict(self, X):
        # Ensure that the number of classifiers (self.k) matches the expected number of classes
        self.k = len(self.clfs)  # Update self.k to the actual number of classifiers

        # Initialize the predictions array
        preds = np.zeros((X.shape[0], self.k))

        # Loop over each classifier and store predictions
        for i, clf in enumerate(self.clfs):
            _, preds[:, i] = clf.predict(X)

        # Return the argmax predictions
        return np.argmax(preds, axis=1)

    def evaluate(self, X, y):
        outputs = self.predict(X)
        accuracy = np.sum(outputs == y) / len(y)
        return round(accuracy, 2)

    def predict(self, X_t):
        # X_t = to_cuda(X_t)
        if self.multiclass: return self.multi_predict(X_t)
        xₛ, yₛ = self.X[self.margin_sv, np.newaxis], self.y[self.margin_sv]
        αs, y, X = self.αs[self.is_sv], self.y[self.is_sv], self.X[self.is_sv]

        b = yₛ - np.sum(αs * y * self.kernel(X, xₛ, self.k), axis=0)
        score = np.sum(αs * y * self.kernel(X, X_t, self.k), axis=0) + b
        return np.sign(score).astype(int), score
