using NeuralNetworks.ActivationFunctions;
using NeuralNetworks.ConsoleSamples.Extensions;
using NeuralNetworks.ConsoleSamples.Helpers;
using NeuralNetworks.CostFunctions;
using NeuralNetworks.Tools;
using NeuralNetworks.Training;
using System;
using System.Collections.Generic;

namespace NeuralNetworks.ConsoleSamples.Examples
{
    public class DigitsRecognitionExample : IExample
    {
        private readonly List<TrainingElement> _trainingData;
        private readonly List<TrainingElement> _validationData;
        private readonly int _epochs;

        public DigitsRecognitionExample(int epochs)
        {
            Console.WriteLine("Initializing data");
            _trainingData = TrainingDataLoader.Load("NeuralNetworks.ConsoleSamples.Resources.digits-image-training-set.json");
            _validationData = TrainingDataLoader.Load("NeuralNetworks.ConsoleSamples.Resources.digits-image-validation-set.json");
            _epochs = epochs;
        }

        public void Run()
        {
            Console.WriteLine("Building random neural network");
            var neuralNetworkDto = Builder.Build(new List<int> { 784, 30, 10 }, ActivationFunctionType.Sigmoid);
            Randomiser.Randomise(neuralNetworkDto);

            Console.WriteLine("Evaluating untrained neural network");
            var untrainedAccuracy = Statistics.GetAccuracyByMax(_validationData, new NeuralNetwork(neuralNetworkDto.ToLayers(), neuralNetworkDto.InputNeuronsCount));
            Console.WriteLine($"Untrained network accuracy: {untrainedAccuracy.ToString("N2")}%");
                        
            var stochasticGradientDescent = new StochasticGradientDescent(new Sigmoid(), new CrossEntropy(), 1, 20, 1D, 0D);

            var maxAccuracy = 0D;

            for (int i = 0; i < _epochs; i++)
            {
                Console.WriteLine($"Epoch {i + 1} started");
                var trainingLength = Statistics.GetTrainingLength(stochasticGradientDescent, neuralNetworkDto, _trainingData);
                var trainingAccuracy = Statistics.GetAccuracyByMax(_validationData, new NeuralNetwork(neuralNetworkDto.ToLayers(), neuralNetworkDto.InputNeuronsCount));
                Console.WriteLine($"Results after epoch {i + 1}:");
                Console.WriteLine($"Training length in miliseconds: {trainingLength}, Accuracy: {trainingAccuracy.ToString("N2")}%");

                if(maxAccuracy < trainingAccuracy)
                {
                    maxAccuracy = trainingAccuracy;
                }
            }

            Console.WriteLine($"End of training. Best accuracy {maxAccuracy.ToString("N2")}%");
        }
    }
}
