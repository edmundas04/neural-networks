using NeuralNetworks.ActivationFunctions;
using NeuralNetworks.ConsoleSamples.Builders;
using NeuralNetworks.ConsoleSamples.Helpers;
using NeuralNetworks.Running;
using NeuralNetworks.Training;

namespace NeuralNetworks.ConsoleSamples.Examples
{
    public class DigitsRecognitionExample : IExample
    {
        public NeuralNetwork CreateNeuralNetwork()
        {
            var result = new NeuralNetworkBuilder().AddNeuronLayer(784, 0.0, 1.0).AddNeuronLayer(30, 1.0, 1.0).AddNeuronLayer(10, 1.0, 1.0).Build();
            NeuralNetworkRandomiser.Randomise(result, 1D);
            return result;
        }
        
        public void Train(NeuralNetwork neuralNetwork)
        {
            var trainingData = TrainingDataLoader.Load("NeuralNetworks.ConsoleSamples.Resources.digits-image-validation-set.json");
            new StochasticGradientDescent(new Sigmoid(), new NeuralNetworkRunner(), 2, 20, 3.0).Train(neuralNetwork, trainingData);
        }

        public void DisplayEvaluation(NeuralNetwork neuralNetwork)
        {
        }
    }
}
