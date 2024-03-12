#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>

class SimpleAnn {
private:
    std::vector<Eigen::MatrixXd> weights;
    std::vector<Eigen::VectorXd> biases;
    std::vector<Eigen::VectorXd> layerOutputs;
    std::vector<Eigen::VectorXd> layerInputs;

public:
    SimpleAnn(const std::vector<int>& layerSizes) {
        for (size_t i = 0; i < layerSizes.size() - 1; ++i) {
            weights.push_back(Eigen::MatrixXd::Random(layerSizes[i + 1], layerSizes[i]));
            biases.push_back(Eigen::VectorXd::Random(layerSizes[i + 1]));
        }
    }

    static double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    static double sigmoid_derivative(double x) {
        return x * (1 - x);
    }

    void forwardPropagate(const Eigen::VectorXd& input) {
        Eigen::VectorXd activation = input;
        layerOutputs.clear();
        layerInputs.clear();
        for (size_t i = 0; i < weights.size(); ++i) {
            Eigen::VectorXd z = weights[i] * activation + biases[i];
            activation = z.unaryExpr(&SimpleAnn::sigmoid);
            layerOutputs.push_back(activation);
            layerInputs.push_back(z);
        }
    }

    void backPropagate(const Eigen::VectorXd& input, const Eigen::VectorXd& target, double learningRate) {
        std::vector<Eigen::VectorXd> errors(weights.size()); // Initialize errors vector
        Eigen::VectorXd outputError = (getOutput() - target).cwiseProduct(layerOutputs.back().unaryExpr(&SimpleAnn::sigmoid_derivative));
        errors[weights.size() - 1] = outputError;

        // Back propagate the error through the network
        for (int i = weights.size() - 2; i >= 0; --i) {
            Eigen::VectorXd error = (weights[i + 1].transpose() * errors[i + 1]).cwiseProduct(layerOutputs[i].unaryExpr(&SimpleAnn::sigmoid_derivative));
            errors[i] = error;
        }

        // Update weights and biases with the learning rate
        for (size_t i = 0; i < weights.size(); ++i) {
            Eigen::VectorXd inputToLayer = (i == 0) ? input : layerOutputs[i - 1];
            Eigen::MatrixXd deltaWeights = errors[i] * inputToLayer.transpose();
            weights[i] -= learningRate * deltaWeights; // Apply learning rate here
            biases[i] -= learningRate * errors[i];
        }
    }


    Eigen::VectorXd getOutput() const {
        if (!layerOutputs.empty()) {
            return layerOutputs.back();
        }
        return Eigen::VectorXd();
    }

    // Prediction method for binary classification
    int predict(const Eigen::VectorXd& input, double threshold = 0.5) {
        forwardPropagate(input);
        Eigen::VectorXd output = getOutput();
        // Assuming a single output neuron for binary classification
        return output(0) > threshold ? 1 : 0;
    }

    double train(const std::vector<Eigen::VectorXd>& inputs, const std::vector<Eigen::VectorXd>& targets, int epochs, double learningRate) {
        double totalLoss = 0.0;
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double epochLoss = 0.0;
            for (size_t i = 0; i < inputs.size(); ++i) {
                forwardPropagate(inputs[i]);
                Eigen::VectorXd output = getOutput();
                Eigen::VectorXd error = output - targets[i];
                epochLoss += error.squaredNorm();
                backPropagate(inputs[i], targets[i], learningRate);
            }
            epochLoss /= inputs.size();
            totalLoss += epochLoss;
            if (epoch % 100 == 0) {
                std::cout << "Epoch: " << epoch << ", Loss: " << epochLoss << std::endl;
            }
        }
        return totalLoss / epochs;
    }

    void evaluate(const std::vector<Eigen::VectorXd>& inputs, const std::vector<Eigen::VectorXd>& targets) {
        int correctPredictions = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            int prediction = predict(inputs[i]);
            int actual = static_cast<int>(targets[i](0)); // Assuming binary targets
            if (prediction == actual) {
                correctPredictions++;
            }
        }
        double accuracy = static_cast<double>(correctPredictions) / inputs.size();
        std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;
    }
};

int main() {
    std::vector<Eigen::VectorXd> inputs = {
            (Eigen::VectorXd(2) << 0, 0).finished(),
            (Eigen::VectorXd(2) << 0, 1).finished(),
            (Eigen::VectorXd(2) << 1, 0).finished(),
            (Eigen::VectorXd(2) << 1, 1).finished()
    };

    std::vector<Eigen::VectorXd> targets = {
            (Eigen::VectorXd(1) << 0).finished(),
            (Eigen::VectorXd(1) << 1).finished(),
            (Eigen::VectorXd(1) << 1).finished(),
            (Eigen::VectorXd(1) << 0).finished()
    };

    SimpleAnn ann({2, 4, 1});
    ann.train(inputs, targets, 10000, 0.1); // Train for 1000 epochs with a learning rate of 0.1

    ann.evaluate(inputs, targets); // Evaluate the trained network

    return 0;
}
