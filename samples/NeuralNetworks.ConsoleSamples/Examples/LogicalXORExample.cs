﻿using NeuralNetworks.ActivationFunctions;
using NeuralNetworks.ConsoleSamples.Helpers;
using NeuralNetworks.Training;
using System;
using System.Collections.Generic;

namespace NeuralNetworks.ConsoleSamples.Examples
{
    public class LogicalXORExample : IExample
    {
        public NeuralNetworkDto CreateNeuralNetwork()
        {
            var result = NeuralNetworkDtoBuilder.Build(new List<int> { 2, 3, 1 }, ActivationFunctionType.Sigmoid);
            NeuralNetworkRandomiser.Randomise(result, 1D);
            return result;
        }

        /// <summary>
        /// In this example neural network is trained to perform logical XOR operation.
        /// </summary>
        /// <returns></returns>
        public void Train(NeuralNetworkDto dto)
        {
            var stochasticGradientDescent = new StochasticGradientDescent(new Sigmoid(),  3000, 4, 5D);
            var trainingData = new List<TrainingElement>
            {
                new TrainingElement
                {
                    Inputs = new double[] { 0D, 0D },
                    ExpectedOutputs = new double[] { 0D }
                },
                new TrainingElement
                {
                    Inputs = new double[] { 1D, 0D },
                    ExpectedOutputs = new double[] { 1D }
                },
                new TrainingElement
                {
                    Inputs = new double[] { 0D, 1D },
                    ExpectedOutputs = new double[] { 1D }
                },
                new TrainingElement
                {
                    Inputs = new double[] { 1D, 1d },
                    ExpectedOutputs = new double[] { 0D }
                }
            };
            
            stochasticGradientDescent.Train(dto, trainingData);
        }

        public  void DisplayEvaluation(NeuralNetworkDto dto)
        {
            var neuralNetwork = new NeuralNetwork(dto);
            var output1 = neuralNetwork.Run(new double[] { 0D, 0D })[0].ToString("N10");
            var output2 = neuralNetwork.Run(new double[] { 1D, 0D })[0].ToString("N10");
            var output3 = neuralNetwork.Run(new double[] { 0D, 1D })[0].ToString("N10");
            var output4 = neuralNetwork.Run(new double[] { 1D, 1D })[0].ToString("N10");
            Console.WriteLine($"Input: 0;0   Output: {output1}   Expected output: 0");
            Console.WriteLine($"Input: 1;0   Output: {output2}   Expected output: 1");
            Console.WriteLine($"Input: 0;1   Output: {output3}   Expected output: 1");
            Console.WriteLine($"Input: 1;1   Output: {output4}   Expected output: 0");
            Console.WriteLine();
        }
    }
}
