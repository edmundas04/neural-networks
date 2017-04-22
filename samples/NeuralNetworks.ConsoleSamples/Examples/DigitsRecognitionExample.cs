using NeuralNetworks.ActivationFunctions;
using NeuralNetworks.ConsoleSamples.Helpers;
using NeuralNetworks.CostFunctions;
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
            var neuralNetworkDto = NeuralNetworkDtoBuilder.Build(new List<int> { 784, 30, 10 }, ActivationFunctionType.Sigmoid);
            NeuralNetworkRandomiser.Randomise(neuralNetworkDto, 1D);

            Console.WriteLine("Evaluating untrained neural network");
            var untrainedAccuracy = StatsCalculator.GetAccuracyByMax(_validationData, neuralNetworkDto);
            Console.WriteLine($"Untrained network accuracy: {untrainedAccuracy}");
                        
            var stochasticGradientDescent = new StochasticGradientDescent(new Sigmoid(), new CrossEntropy(), 1, 20, 1D);

            var maxAccuracy = 0D;

            for (int i = 0; i < _epochs; i++)
            {
                Console.WriteLine($"Epoch {i + 1} started");
                var trainingLength = StatsCalculator.GetTrainingLength(stochasticGradientDescent, neuralNetworkDto, _trainingData);
                var trainingAccuracy = StatsCalculator.GetAccuracyByMax(_validationData, neuralNetworkDto);
                Console.WriteLine($"Results after epoch {i + 1}:");
                Console.WriteLine($"Training length in miliseconds: {trainingLength}, Accuracy: {trainingAccuracy.ToString("N2")}");

                if(maxAccuracy < trainingAccuracy)
                {
                    maxAccuracy = trainingAccuracy;
                }
            }

            Console.WriteLine($"End of training. Best accuracy {maxAccuracy}");
        }
    }
}
