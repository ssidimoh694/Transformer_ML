import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import math
from torchinfo import summary
from src.data import load_data
from src.methods.pca import PCA
from src.methods.deep_network import MLP, CNN, Trainer, MyViT
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes
import random

#---------------------------------------MAIN---------------------------------------
def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors (60000x784)
    xtrain, xtest, ytrain = load_data(args.data)    
    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.
    # Make a validation set
    if not args.test:
    ### WRITE YOUR CODE HERE
        ### WRITE YOUR CODE HERE
        N=xtrain.shape[0]
        split_ratio = 0.8
        indices = np.arange(N)
        np.random.shuffle(indices)
        x_train = xtrain
        y_train = ytrain

        x_train    = xtrain[indices[0:int(N*split_ratio)],:]
        y_train    = ytrain[indices[0:int(N*split_ratio)]]
       
        xtest     = xtrain[indices[int(N*(split_ratio)):],:]
        ytest     = ytrain[indices[int(N*split_ratio):]]

    ### WRITE YOUR CODE HERE to do any other data processing

    # Dimensionality reduction (MS2)
    if args.use_pca:

        print("Using PCA")
        pca_obj = PCA(d=args.pca_d)
        pca_obj.find_principal_components(xtrain)
        xtrain = pca_obj.reduce_dimension(xtrain)

    ## 3. Initialize the method you want to use.

    # Neural Networks (MS2)

    # Prepare the model (and data) for Pytorch
    # Note: you might need to reshape the data depending on the network you use!
    n_classes = get_n_classes(ytrain)
    if args.nn_type == "mlp":
        model = ... ### WRITE YOUR CODE HERE
    random_ints = [1,2,3,4]
    hidden_d_vals = [6, 8, 10, 9, 12, 14]
    n_blocks_vals = [2, 3, 4, 5]
    n_heads_vals = [2, 3, 4]
    lr_vals = [1e-4, 3e-4, 5e-4, 7e-4, 9e-4]

    if args.nn_type == "transformer":
        x_train = np.reshape(x_train, (x_train.shape[0], 1, 28, 28))
        xtest = np.reshape(xtest, (xtest.shape[0], 1, 28, 28))
        for i in range(30):
            
            n_blocks = random.choice(n_blocks_vals)
            hidden_d = random.choice(hidden_d_vals)
            while True:
                n_heads = random.choice(n_heads_vals)
                if hidden_d % n_heads == 0:
                    break
            lr = random.choice(lr_vals)

            print(f"Run {i+1}:")
            print(f"  hidden_d: {hidden_d}, n_blocks: {n_blocks}, n_heads: {n_heads}, lr: {lr}")
            model = MyViT((1, 28, 28), n_patches=7, n_blocks=n_blocks, hidden_d=hidden_d, n_heads=n_heads, out_d=n_classes)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            method_obj = Trainer(model, criterion, optimizer, epochs=15)
            
            # Entraîner le modèle
            method_obj.fit(x_train, y_train)
            
            # Prédictions et calcul de l'accuracy
            preds_test = method_obj.predict(xtest)
            accuracy = accuracy_fn(preds_test, ytest)

            print(f"  Accuracy: {accuracy}")
    exit(0)
    # Trainer object

    ## 4. Train and evaluate the method

    # Fit (:=train) the method on the training data
    preds_train = method_obj.fit(xtrain, ytrain)

    # Predict on unseen data
    preds = method_obj.predict(xtest)

    ## Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


    ## As there are no test dataset labels, check your model accuracy on validation dataset.
    # You can check your model performance on test set by submitting your test set predictions on the AIcrowd competition.
    acc = accuracy_fn(preds, xtest)
    macrof1 = macrof1_fn(preds, xtest)
    print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    # Feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument('--data', default="dataset", type=str, help="path to your dataset")
    parser.add_argument('--nn_type', default="mlp",
                        help="which network architecture to use, it can be 'mlp' | 'transformer' | 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")
    parser.add_argument('--device', type=str, default="cpu",
                        help="Device to use for the training, it can be 'cpu' | 'cuda' | 'mps'")
    parser.add_argument('--use_pca', action="store_true", help="use PCA for feature reduction")
    parser.add_argument('--pca_d', type=int, default=100, help="the number of principal components")


    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true",
                        help="train on whole training data and evaluate on the test data, otherwise use a validation set")


    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)
