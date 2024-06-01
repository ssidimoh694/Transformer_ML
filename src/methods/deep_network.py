import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math
## MS2


class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_size, n_classes, my_arg=32)
        
        Arguments:
            input_size (int): size of the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, D)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        return preds


class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)
        
        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        return preds

""" 
    Patchify the images into n_patches x n_patches patches.
    Arguments:
        images (tensor): input batch of shape (N, Ch, H, W) | H = W
        n_patches (int): number of patches in each dimension
    Returns:
        patches (tensor): patches of shape (N, n_patches ** 2, patch_size ** 2 * Ch)
    """
#IMPORTANT NOTE : The slow (patchify, pos_embeddings and MSA) class and function are my own functions that I coded mySelf. I helped my self with chatgpt
#to write faster and more optimal functions to have a faster training time and to better optimise the transformer Model. 
def patchify(images, n_patches):
    n, c, h, w = images.shape
    assert h == w, "Images must be square"

    patch_size = h // n_patches
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.view(n, n_patches ** 2, -1).float()  # Convert patches to float

    return patches

def slowPatchify(images, n_patches):
    n, c, h, w = images.shape
    assert h == w # We assume square image.
    
    patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)

    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):

                # Extract the patch of the image.
                patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size] ### WRITE YOUR CODE HERE

                # Flatten the patch and store it.
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches

"""
Get the positional embeddings for the patches.
Arguments:
    sequence_length (int): number of patches
    d (int): hidden dimension
Returns:
    result (tensor): positional embeddings of shape (sequence_length, d)
"""
def slow_get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(int(d/2)):
            result[i, 2*j] = math.sin(i / (10000 ** (2 * j / d)))
            result[i, 2*j + 1] = math.cos(i / (10000 ** (2 * j / d)))
    return result 

  def get_positional_embeddings(sequence_length, d):
    result = torch.zeros(sequence_length, d)
    positions = torch.arange(sequence_length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
    
    result[:, 0::2] = torch.sin(positions * div_term)
    if d % 2 == 1:
        result[:, 1::2] = torch.cos(positions * div_term[:-1])
    else:
        result[:, 1::2] = torch.cos(positions * div_term)

    return result

class slowMyMSA(nn.Module):
    """
    Multi-Head Self Attention Module
    """
    def __init__(self, d, n_heads=2):

        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"
        d_head = int(d / n_heads)
        self.d_head = d_head

        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):

                # Select the mapping associated to the given head.
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]

                # Map seq to q, k, v.
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq) ### WRITE YOUR CODE HERE
                
                attention = self.softmax(torch.matmul(q,k.T/math.sqrt(sequence.shape[1])))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])
    
class MyMSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads
        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"
        self.d_head = d // n_heads

        self.q_mappings = nn.Linear(d, d)
        self.k_mappings = nn.Linear(d, d)
        self.v_mappings = nn.Linear(d, d)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        batch_size, seq_length, _ = sequences.size()
        q = self.q_mappings(sequences).view(batch_size, seq_length, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_mappings(sequences).view(batch_size, seq_length, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_mappings(sequences).view(batch_size, seq_length, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attention = self.softmax(scores)
        context = torch.matmul(attention, v).transpose(1, 2).contiguous().view(batch_size, seq_length, self.d)

        return context


class MyViTBlock(nn.Module):
    """
    Transformer block for the Vision Transformer.
    """
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(hidden_d, n_heads)
        self.norm2 =nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        # Write code for MHSA + residual connection.
        out = x + self.mhsa(self.norm1(x))
        # Write code for MLP(Norm(out)) + residual connection
        out = out + self.mlp(out)
        return out

class MyViT(nn.Module):
    def __init__(self, chw, n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10):
        super(MyViT, self).__init__()

        self.chw = chw # (C, H, W)
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d

        # Input and patches sizes
        assert chw[1] % n_patches == 0 # Input shape must be divisible by number of patches
        assert chw[2] % n_patches == 0
        self.patch_size =  (chw[1] / n_patches, chw[2] / n_patches)

        # Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # Positional embedding
        # HINT: don't forget the classification token
        self.positional_embeddings =  get_positional_embeddings(n_patches ** 2 + 1, hidden_d)

        # Transformer blocks
        self.blocks = nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])

        # Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        n, _, _, _= x.shape

        # Divide images into patches.
        patches = patchify(x, self.n_patches) ### WRITE YOUR CODE HERE

        # Map the vector corresponding to each patch to the hidden size dimension.
        tokens = self.linear_mapper(patches) ### WRITE YOUR CODE HERE

        # Add classification token to the tokens.
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        # Add positional embedding.
        # HINT: use torch.Tensor.repeat(...)
        out =  out =  tokens + self.positional_embeddings.repeat(n, 1, 1)

        # Transformer Blocks
        for block in self.blocks:
            out = block(out)

        # Get the classification token only.
        out = out[:, 0]

        # Map to the output distribution.
        out = self.mlp(out)

        return out


class Trainer(object):
    """
    Trainer class for the deep networks.

    It will also serve as an interface between numpy and pytorch.
    """

    def __init__(self, model, criterion, optimizer, epochs=20, batch_size=128):
        """
        Initialize the trainer object for a given model.

        Arguments:
            model (nn.Module): the model to train
            lr (float): learning rate for the optimizer
            epochs (int): number of epochs of training
            batch_size (int): number of data points in each batch
            optimizer: The optimizer to use.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size

    """
    Compute the accuracy of the model on the data.
    Arguments:
        x (tensor): input batch of shape (N, D)
        y (tensor): target batch of shape (N,)
    Returns:
        accuracy (float): accuracy of the model on the data
    """
    def accuracy(self, x, y):
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        return np.mean(np.argmax(x, axis=1) == y)
    
    def train_all(self, dataloader):
        """
        Fully train the model over the epochs. 
        
        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        self.model.train()
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader)
    def train_one_epoch(self, dataloader):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        for it, (x,y) in enumerate(dataloader):
            # Load a batch, break it down in images and targets.
            # Run forward pass.
            logits = self.model(x)
            # Compute loss (using 'criterion').
            loss = self.criterion(logits, y)
            # Run backward pass.
            loss.backward()

            # Update the weights using 'optimizer'.
            self.optimizer.step()

            # Zero-out the accumulated gradients.
            self.optimizer.zero_grad()
            #print('it {}/{}  '.format(it + 1, len(self.dataloader_train)), end='')


    def predict_torch(self, dataloader):
        """
        Predict the validation/test dataloader labels using the model.

        Hints:
            1. Don't forget to set your model to eval mode, i.e., self.model.eval()!
            2. You can use torch.no_grad() to turn off gradient computation, 
            which can save memory and speed up computation. Simply write:
                with torch.no_grad():
                    # Write your code here.

        Arguments:
            dataloader (DataLoader): dataloader for validation/test data
        Returns:
            pred_labels (torch.tensor): predicted labels of shape (N,),
                with N the number of data points in the validation/test data.
        """

        self.model.eval()  # Set the model to evaluation mode
        pred_labels = np.array([])

        with torch.no_grad():  # Disable gradient calculations
            for inputs in dataloader:
                predictions = self.model(inputs[0])
                # Get the predicted class for each example in the batch
                predicted_classes = np.argmax(predictions, axis=1)
                # Check which predictions are correct
                # Filter the correct predictions
                pred_labels = np.append(pred_labels, predicted_classes)

        return pred_labels

    
    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
           
             pred_labels (array): target of shape (N,)
        """
        # First, prepare data for pytorch
        train_dataset = TensorDataset(torch.from_numpy(training_data).float(), 
                                      torch.from_numpy(training_labels).long())
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        self.train_all(train_dataloader)

        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        This serves as an interface between numpy and pytorch.
        
        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        # First, prepare data for pytorch
        test_dataset = TensorDataset(torch.from_numpy(test_data).float())
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        pred_labels = self.predict_torch(test_dataloader)

        # We return the labels after transforming them into numpy array.
        return pred_labels
