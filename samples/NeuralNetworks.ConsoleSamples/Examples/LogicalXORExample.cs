using NeuralNetworks.ActivationFunctions;
using NeuralNetworks.CostFunctions;
using NeuralNetworks.Layers;
using NeuralNetworks.Tools;
using NeuralNetworks.Training;
using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworks.ConsoleSamples.Examples
{
    public class LogicalXORExample : IExample
    {

        public void Run()
        {
            var layer = new ILayer[] { new FullyConnectedLayer(new Sigmoid(), 3, 2), new FullyConnectedLayer(new Sigmoid(), 1, 3) };
            Randomiser.Randomise(layer, new Random(5));

            Console.WriteLine("Evaluationg untrained neural network");
            DisplayEvaluation(layer);

            var stochasticGradientDescent = new StochasticGradientDescent(new CrossEntropy(), layer, 3000, 4, 5D, 0D);
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

            stochasticGradientDescent.Train(trainingData);

            Console.WriteLine("Evaluationg trained neural network");
            DisplayEvaluation(layer);
        }

        private void DisplayEvaluation(ILayer[] layers)
        {
            var neuralNetwork = new NeuralNetwork(layers, layers.First().PrimaryNeuronsCount);
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
