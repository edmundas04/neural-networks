using NeuralNetworks.ActivationFunctions;
using NeuralNetworks.ConsoleSamples.Helpers;
using NeuralNetworks.CostFunctions;
using NeuralNetworks.Tools;
using NeuralNetworks.Training;
using System;
using System.Collections.Generic;

namespace NeuralNetworks.ConsoleSamples.Examples
{
    public class DigitsRecognitionCompareExample : IExample
    {
        private readonly List<TrainingElement> _trainingData;
        private readonly List<TrainingElement> _validationData;
        private readonly int _epochs;

        public DigitsRecognitionCompareExample(int epochs)
        {
            Console.WriteLine("Initializing data");
            _trainingData = TrainingDataLoader.Load("NeuralNetworks.ConsoleSamples.Resources.digits-image-training-set.json");
            _validationData = TrainingDataLoader.Load("NeuralNetworks.ConsoleSamples.Resources.digits-image-validation-set.json");
            _epochs = epochs;
        }

        public void Run()
        {
            var neuralNetworkDto = Builder.Build(new List<int> { 784, 30, 10 }, ActivationFunctionType.Sigmoid);
            Randomiser.Randomise(neuralNetworkDto);

            
            var trainersToCompare = new List<ITrainer>
            {
                new StochasticGradientDescent(new Sigmoid(), new Quadratic(), _epochs, 10, 1D, 0),
                new StochasticGradientDescent(new Sigmoid(), new Quadratic(), _epochs, 20, 1D, 0),
                new StochasticGradientDescent(new Sigmoid(), new Quadratic(), _epochs, 20, 1D, 0),
                new StochasticGradientDescent(new Sigmoid(), new CrossEntropy(), _epochs, 10, 1D, 0),
                new StochasticGradientDescent(new Sigmoid(), new CrossEntropy(), _epochs, 20, 1D, 0),
                new StochasticGradientDescent(new Sigmoid(), new CrossEntropy(), _epochs, 30, 1D, 0)
            };

            Console.WriteLine("Comparison started");
            var compareResults = TrainersComparer.Compare(trainersToCompare, neuralNetworkDto, _trainingData, _validationData);
            Console.WriteLine("Comparison ended");

            Console.WriteLine("Results");
            for (int i = 0; i < compareResults.Count; i++)
            {
                var compareResult = compareResults[i];
                Console.WriteLine($"{i + 1}. Time ellapsed: {compareResult.ElapsedTimeInMillisecond}, Accuracy: {compareResult.Accuracy.ToString("N2")}%, AccuracyByMax: {compareResult.AccuracyByMax.ToString("N2")}%");
            }
        }
    }
}
