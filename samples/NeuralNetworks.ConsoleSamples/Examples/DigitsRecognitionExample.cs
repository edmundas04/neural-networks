using NeuralNetworks.ActivationFunctions;
using NeuralNetworks.ConsoleSamples.Builders;
using NeuralNetworks.ConsoleSamples.Helpers;
using NeuralNetworks.Running;
using NeuralNetworks.Training;
using System;
using System.Collections.Generic;
using System.Linq;

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
            var trainingData = TrainingDataLoader.Load("NeuralNetworks.ConsoleSamples.Resources.digits-image-training-set.json");
            var stochasticGradientDescent = new StochasticGradientDescent(new Sigmoid(), 2, 20, 3D);
            stochasticGradientDescent.Train(neuralNetwork, trainingData);
        }

        public void DisplayEvaluation(NeuralNetwork neuralNetwork)
        {
            var validationData = TrainingDataLoader.Load("NeuralNetworks.ConsoleSamples.Resources.digits-image-validation-set.json");
            var neuralNetworkRunner = new NeuralNetworkRunner();

            var correctCount = 0;

            foreach (var validationItem in validationData)
            {
                var output = neuralNetworkRunner.Run(neuralNetwork, validationItem.Inputs.ToList());
                if (CheckOutput(output, validationItem.ExpectedOutputs.ToList()))
                {
                    correctCount++;
                }
            }

            Console.WriteLine($"Correctly recognized {correctCount} digits out of {validationData.Count}");
        }

        public bool CheckOutput(List<double> output, List<double> expectedOutput)
        {
            var maxIndex = 0;
            var maxValue = output[0];

            for (int i = 1; i < output.Count; i++)
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
