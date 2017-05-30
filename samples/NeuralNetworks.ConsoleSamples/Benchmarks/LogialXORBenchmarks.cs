using BenchmarkDotNet.Attributes;
using NeuralNetworks.ActivationFunctions;
using NeuralNetworks.CostFunctions;
using NeuralNetworks.Layers;
using NeuralNetworks.Tools;
using NeuralNetworks.Training;
using System;
using System.Collections.Generic;

namespace NeuralNetworks.ConsoleSamples.Benchmarks
{
    public class LogialXORBenchmarks
    {
        private readonly ITrainer _trainer;

        private readonly List<TrainingElement> _trainingData;

        public LogialXORBenchmarks()
        {
            _trainingData = new List<TrainingElement>
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
            
            var layer = new ILayer[] { new FullyConnectedLayer(new Sigmoid(), 3, 2), new FullyConnectedLayer(new Sigmoid(), 1, 3) };
            Randomiser.Randomise(layer, new Random(1));

            _trainer = new StochasticGradientDescent(new CrossEntropy(), layer, 1000, 4, 5D, 0D);
            
        }

        [Benchmark]
        public void StochasticGradientDescentNewBenchmark()
        {
            _trainer.Train(_trainingData);
        }
    }
}
