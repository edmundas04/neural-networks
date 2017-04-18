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
        public NeuralNetworkDto CreateNeuralNetwork()
        {
            var result = NeuralNetworkDtoBuilder.Build(new List<int> { 784, 30, 10 }, ActivationFunctionType.Sigmoid);
            NeuralNetworkRandomiser.Randomise(result, 1D);
            return result;
        }
        
        public void Train(NeuralNetworkDto neuralNetwork)
        {
            var trainingData = TrainingDataLoader.Load("NeuralNetworks.ConsoleSamples.Resources.digits-image-training-set.json");
            var stochasticGradientDescent = new StochasticGradientDescent(new Sigmoid(), new Quadratic(), 2, 20, 3D);
            stochasticGradientDescent.Train(neuralNetwork, trainingData);
        }

        public void DisplayEvaluation(NeuralNetworkDto dto)
        {
            var validationData = TrainingDataLoader.Load("NeuralNetworks.ConsoleSamples.Resources.digits-image-validation-set.json");
            var neuralNetwork = new NeuralNetwork(dto);

            var correctCount = 0;

            foreach (var validationItem in validationData)
            {
                var output = neuralNetwork.Run(validationItem.Inputs);
                if (CheckOutput(output, validationItem.ExpectedOutputs))
                {
                    correctCount++;
                }
            }

            Console.WriteLine($"Correctly recognized {correctCount} digits out of {validationData.Count}");
        }

        public bool CheckOutput(double[] output, double[] expectedOutput)
        {
            var maxIndex = 0;
            var maxValue = output[0];

            for (int i = 1; i < output.Length; i++)
            {
                if(output[i] > maxValue)
                {
                    maxValue = output[i];
                    maxIndex = i;
                }
            }

            return expectedOutput[maxIndex] == 1D;
        }
    }
}
