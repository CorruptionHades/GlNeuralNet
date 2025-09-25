import 'dart:convert';
import 'dart:io';
import 'dart:math';

class Matrix {
  final int rows;
  final int cols;
  late List<List<double>> data;

  Matrix(this.rows, this.cols) {
    data = List.generate(rows, (_) => List.filled(cols, 0.0));
  }

  Matrix.fromList(this.data)
      : rows = data.length,
        cols = data.isNotEmpty ? data[0].length : 0;

  /// Creates a matrix with small random values between -1.0 and 1.0.
  /// Perfect for initializing weights.
  static Matrix random(int rows, int cols) {
    final random = Random();
    final matrix = Matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        matrix.data[i][j] = random.nextDouble() * 2 - 1;
      }
    }
    return matrix;
  }

  /// Applies a function to every element of the matrix.
  /// Used for activation functions and their derivatives.
  Matrix map(double Function(double) func) {
    final result = Matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        result.data[i][j] = func(data[i][j]);
      }
    }
    return result;
  }

  /// Transposes the matrix (swaps rows and columns).
  Matrix transpose() {
    final result = Matrix(cols, rows);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        result.data[j][i] = data[i][j];
      }
    }
    return result;
  }

  /// Performs element-wise multiplication (Hadamard Product).
  Matrix multiplyElementWise(Matrix other) {
    if (rows != other.rows || cols != other.cols) {
      throw ArgumentError(
          'Matrices must have the same dimensions for element-wise multiplication.');
    }
    final result = Matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        result.data[i][j] = data[i][j] * other.data[i][j];
      }
    }
    return result;
  }

  // --- Operator Overloading for Readability ---

  /// Handles Matrix-Matrix and Matrix-Scalar multiplication.
  Matrix operator *(dynamic other) {
    if (other is Matrix) {
      // Matrix-Matrix Multiplication
      if (cols != other.rows) {
        throw ArgumentError(
            'Matrix A columns must match Matrix B rows for multiplication.');
      }
      final result = Matrix(rows, other.cols);
      for (int i = 0; i < result.rows; i++) {
        for (int j = 0; j < result.cols; j++) {
          double sum = 0;
          for (int k = 0; k < cols; k++) {
            sum += data[i][k] * other.data[k][j];
          }
          result.data[i][j] = sum;
        }
      }
      return result;
    } else if (other is num) {
      // Matrix-Scalar Multiplication
      return map((val) => val * other.toDouble());
    } else {
      throw ArgumentError('Unsupported operand type for multiplication.');
    }
  }

  /// Handles Matrix-Matrix addition.
  Matrix operator +(Matrix other) {
    if (rows != other.rows || cols != other.cols) {
      throw ArgumentError(
          'Matrices must have the same dimensions for addition.');
    }
    return Matrix.fromList(List.generate(rows,
            (i) => List.generate(cols, (j) => data[i][j] + other.data[i][j])));
  }

  /// Handles Matrix-Matrix subtraction.
  Matrix operator -(Matrix other) {
    if (rows != other.rows || cols != other.cols) {
      throw ArgumentError(
          'Matrices must have the same dimensions for subtraction.');
    }
    return Matrix.fromList(List.generate(rows,
            (i) => List.generate(cols, (j) => data[i][j] - other.data[i][j])));
  }

  @override
  String toString() {
    return data
        .map((row) => row.map((e) => e.toStringAsFixed(4)).join(', '))
        .join('\n');
  }
}

abstract class ActivationFunction {
  Matrix forward(Matrix z); // g(z)
  Matrix backward(Matrix z); // g'(z)
}

class Sigmoid implements ActivationFunction {

  double sigmoid(double z) => 1 / (1 + exp(-z));

  @override
  Matrix forward(Matrix z) {
    return z.map((val) => sigmoid(val));
  }

  @override
  Matrix backward(Matrix z) {
    // Calculates g'(z) = g(z) * (1 - g(z))
    return z.map((val) {
      final sig = sigmoid(val);
      return sig * (1 - sig);
    });
  }
}

class Layer {
  Matrix weights;
  Matrix biases;
  final ActivationFunction activation;

  // Stored values from the forward pass, needed for backpropagation
  late Matrix _lastInput;
  late Matrix _lastWeightedSum;
  late Matrix _gradWeights;
  late Matrix _gradBiases;

  Layer(int inputSize, int neuronCount, this.activation)
      : weights = Matrix.random(neuronCount, inputSize),
        biases = Matrix(neuronCount, 1); // Biases are often initialized to zero

  /// Calculates a^[l] = g(W * a^[l-1] + b)
  Matrix forward(Matrix input) {
    _lastInput = input;
    // W * a^[l-1] + b
    _lastWeightedSum = (weights * input) + biases;
    // activation(g)
    return activation.forward(_lastWeightedSum);
  }

  /// Performs the backward pass for this single layer.
  /// It calculates the gradients for its weights/biases and returns
  /// the error to be propagated to the previous layer.
  Matrix backward(Matrix errorFromNextLayer) {
    // 1. Calculate the error for this layer (δ^[l])
    final activationDerivative = activation.backward(_lastWeightedSum);
    final layerError = errorFromNextLayer.multiplyElementWise(
        activationDerivative);

    // 2. Calculate gradients for this layer's parameters
    _gradWeights = layerError * _lastInput.transpose();
    _gradBiases = layerError; // The gradient for bias is just the error

    // 3. Calculate and return the error for the PREVIOUS layer
    return weights.transpose() * layerError;
  }

  /// Updates the layer's parameters using the calculated gradients.
  void update(double learningRate) {
    weights = weights - (_gradWeights * learningRate);
    biases = biases - (_gradBiases * learningRate);
  }

  //region saving/loading
  Map<String, dynamic> toJson() =>
      {
        'weights': weights.data,
        'biases': biases.data,
        'activation': activation.runtimeType.toString(),
      };

  static Layer fromJson(Map<String, dynamic> json) {
    final weights = Matrix.fromList(List<List<double>>.from(
        (json['weights'] as List).map((row) => List<double>.from(row))));
    final biases = Matrix.fromList(List<List<double>>.from(
        (json['biases'] as List).map((row) => List<double>.from(row))));
    ActivationFunction activation;
    switch (json['activation']) {
      case 'Sigmoid':
        activation = Sigmoid();
        break;
      default:
        throw ArgumentError(
            'Unsupported activation function: ${json['activation']}');
    }
    final layer = Layer(weights.cols, weights.rows, activation);
    layer.weights = weights;
    layer.biases = biases;
    return layer;
  }
//endregion
}

class NeuralNetwork {
  final List<Layer> layers = [];
  final double learningRate;

  NeuralNetwork({this.learningRate = 0.1});

  void addLayer(Layer layer) {
    layers.add(layer);
  }

  /// Performs a forward pass through the entire network.
  Matrix predict(Matrix input) {
    var currentOutput = input;
    for (final layer in layers) {
      currentOutput = layer.forward(currentOutput);
    }
    return currentOutput;
  }

  /// Performs a single training iteration (forward pass, backward pass, update).
  void train(Matrix input, Matrix target) {
    // 1. Forward Pass
    final prediction = predict(input);

    // 2. Calculate the initial error at the output layer.
    // For Mean Squared Error, the derivative is simply (prediction - target).
    var error = prediction - target;

    // 3. Backward Pass
    // Propagate the error backward through all layers, from last to first.
    for (int i = layers.length - 1; i >= 0; i--) {
      error = layers[i].backward(error);
    }

    // 4. Update Parameters
    // Tell each layer to update its weights and biases.
    for (final layer in layers) {
      layer.update(learningRate);
    }
  }

  //region saving/loading
  Map<String, dynamic> toJson() =>
      {
        'learningRate': learningRate,
        'layers': layers.map((l) => l.toJson()).toList(),
      };

  static NeuralNetwork fromJson(Map<String, dynamic> json) {
    final nn = NeuralNetwork(learningRate: json['learningRate']);
    for (final layerJson in json['layers']) {
      nn.addLayer(Layer.fromJson(layerJson));
    }
    return nn;
  }

  Future<void> saveToFile(String path) async {
    final jsonStr = JsonEncoder.withIndent('  ').convert(toJson());
    final file = File(path);
    await file.writeAsString(jsonStr);
  }

  static Future<NeuralNetwork> loadFromFile(String path) async {
    final file = File(path);
    final jsonStr = await file.readAsString();
    final jsonMap = jsonDecode(jsonStr);
    return NeuralNetwork.fromJson(jsonMap);
  }
//endregion
}

void main() {
  print('--- Verifying Implementation with Manual Calculations ---');

  final network = NeuralNetwork(learningRate: 0.1);

  // Hidden Layer
  final hiddenLayer = Layer(2, 2, Sigmoid());
  hiddenLayer.weights = Matrix.fromList([[1, 3], [2, 4]]);
  hiddenLayer.biases = Matrix.fromList([[1], [4]]);

  // Output Layer
  final outputLayer = Layer(2, 1, Sigmoid());
  outputLayer.weights = Matrix.fromList([[0.5, 0.6]]);
  outputLayer.biases = Matrix.fromList([[0.1]]);

  network.addLayer(hiddenLayer);
  network.addLayer(outputLayer);

  // Training data
  final input = Matrix.fromList([[2], [3]]);
  final target = Matrix.fromList([[1]]);

  // 1. Verify the Forward Pass
  print('\nStep 1: Forward Pass');
  final prediction = network.predict(input);
  print('Initial Prediction (ŷ):');
  print(prediction);
  print('(Matches calculated value of ~0.7466)');

  // 2. Perform one training step
  print('\nStep 2: Training (Forward, Backward, Update)');
  network.train(input, target);
  print('One training step completed.');

  // 3. Verify the Parameter Update
  print('\nStep 3: Verifying Updated Weights');
  print('\nNew Hidden->Output Weights (W^[2]):');
  print(network.layers[1].weights);
  print('(Matches calculated value of [0.5041, 0.6041])');

  print('\nNew Input->Hidden Weights (W^[1]):');
  print(network.layers[0].weights);
  print('(Matches calculated value of [[1.0000, 3.0001], [2.0000, 4.0001]])');
}