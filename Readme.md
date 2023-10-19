# Neural Network Training Script

This Python script is used for training a neural network with a single hidden layer. It demonstrates a simple feedforward neural network using sigmoid activation functions.

## Prerequisites

Make sure you have the necessary Python libraries installed. You can install them using pip:

```bash
pip install numpy argparse
```

## Usage

To run the script, execute the following command:

```bash
python neural_network_training.py --data data.csv --iterations num_iterations --eta learning_rate
```

- `data.csv`: The path to the CSV file containing your dataset. The CSV file should include features and labels.
- `num_iterations`: The number of training iterations or epochs.
- `learning_rate`: The learning rate for gradient descent.

## Algorithm

The script trains a simple neural network with a single hidden layer using a feedforward approach and backpropagation. The script performs the following steps:

1. Read the dataset from the provided CSV file.
2. Initialize weights and biases for the hidden layer and output layer.
3. For each training iteration, forward propagate the input through the neural network.
4. Calculate the error and update weights and biases using backpropagation.
5. Print the results for each iteration, including inputs, hidden layer outputs, output layer outputs, errors, and weights.

## Output

The script will output the training process with detailed information for each iteration, including input values, hidden layer outputs, output layer outputs, errors, and updated weights and biases.

The neural network aims to learn the patterns in the training data to minimize the error and make predictions.

## Example

Here is an example command to run the script:

```bash
python neural_network_training.py --data data.csv --iterations 1000 --eta 0.1
```

This will train the neural network with the data from `data.csv` for 1000 iterations using a learning rate of 0.1.

You can adjust the number of iterations and the learning rate to fine-tune the training process for your specific dataset.
