using NeuralNetworks.ActivationFunctions;
using NeuralNetworks.ConsoleSamples.Builders;
using NeuralNetworks.ConsoleSamples.Helpers;
using NeuralNetworks.Running;
using NeuralNetworks.Training;
using System;
using System.Collections.Generic;

namespace NeuralNetworks.ConsoleSamples.Examples
{
    public class LogicalXORExample : IExample
    {
        public NeuralNetwork CreateNeuralNetwork()
        {
            var result = new NeuralNetworkBuilder()
                .AddNeuronLayer(2, 0, 1)
                .AddNeuronLayer(3, 1, 1)
                .AddNeuronLayer(1, 1, 1)
                .Build();

            NeuralNetworkRandomiser.Randomise(result, 1D);
            return result;
        }

        /// <summary>
        /// In this example neural network is trained to perform logical XOR operation.
        /// </summary>
        /// <returns></returns>
        public void Train(NeuralNetwork neuralNetwork)
        {
            var neuralNetworkRunner = new NeuralNetworkRunner();
            var stochasticGradientDescent = new StochasticGradientDescent(new Sigmoid(),  3000, 4, 5D);
            var trainingData = new List<TrainingElement>
            {
                new TrainingElement
                {
                    Inputs = new List<double> { 0D, 0D },
                    ExpectedOutputs = new List<double> { 0D }
                },
                new TrainingElement
                {
                    Inputs = new List<double> { 1D, 0D },
                    ExpectedOutputs = new List<double> { 1D }
                },
                new TrainingElement
                {
                    Inputs = new List<double> { 0D, 1D },
                    ExpectedOutputs = new List<double> { 1D }
                },
                new TrainingElement
                {
                    Inputs = new List<double> { 1D, 1d },
                    ExpectedOutputs = new List<double> { 0D }
                }
            };
            
            stochasticGradientDescent.Train(neuralNetwork, trainingData);
        }

        public  void DisplayEvaluation(NeuralNetwork neuralNetwork)
        {
            var neuralNetworkRunner = new NeuralNetworkRunner();
            var output1 = neuralNetworkRunner.Run(neuralNetwork, new List<double> { 0D, 0D })[0].ToString("N10");
            var output2 = neuralNetworkRunner.Run(neuralNetwork, new List<double> { 1D, 0D })[0].ToString("N10");
            var output3 = neuralNetworkRunner.Run(neuralNetwork, new List<double> { 0D, 1D })[0].ToString("N10");
            var output4 = neuralNetworkRunner.Run(neuralNetwork, new List<double> { 1D, 1D })[0].ToString("N10");
            Console.WriteLine($"Input: 0;0   Output: {output1}   Expected output: 0");
            Console.WriteLine($"Input: 1;0   Output: {output2}   Expected output: 1");
            Console.WriteLine($"Input: 0;1   Output: {output3}   Expected output: 1");
            Console.WriteLine($"Input: 1;1   Output: {output4}   Expected output: 0");
            Console.WriteLine();
        }
    }
}
