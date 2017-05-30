using BenchmarkDotNet.Attributes;
using NeuralNetworks.ActivationFunctions;
using NeuralNetworks.ConsoleSamples.Extensions;
using NeuralNetworks.CostFunctions;
using NeuralNetworks.Tools;
using NeuralNetworks.Training;
using System;
using System.Collections.Generic;

namespace NeuralNetworks.ConsoleSamples.Benchmarks
{
    public class LogialXORBenchmarks
    {
        private readonly ITrainer _trainer1;
        private readonly NeuralNetworkDto _neuralNetworkDto1;

        private readonly ITrainerNew _trainer2;
        private readonly NeuralNetworkDto _neuralNetworkDto2;

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

            _neuralNetworkDto1 = Builder.Build(new List<int> { 2, 3, 1 }, ActivationFunctionType.Sigmoid);
            Randomiser.Randomise(_neuralNetworkDto1, new Random(1));
            _trainer1 = new StochasticGradientDescent(new Sigmoid(), new CrossEntropy(), 1000, 4, 5D, 0D);

            _neuralNetworkDto2 = Builder.Build(new List<int> { 2, 3, 1 }, ActivationFunctionType.Sigmoid);
            Randomiser.Randomise(_neuralNetworkDto2, new Random(1));
            _trainer2 = new StochasticGradientDescentNew(new CrossEntropy(), _neuralNetworkDto2.ToLayers(), 1000, 4, 5D, 0D);
            
        }

        [Benchmark]
        public void StochasticGradientDescentBenchmark()
        {
            _trainer1.Train(_neuralNetworkDto1, _trainingData);
        }

        [Benchmark]
        public void StochasticGradientDescentNewBenchmark()
        {
            _trainer2.Train(_trainingData);
        }
    }
}
