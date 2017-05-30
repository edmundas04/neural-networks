using BenchmarkDotNet.Attributes;
using NeuralNetworks.ActivationFunctions;
using NeuralNetworks.ConsoleSamples.Helpers;
using NeuralNetworks.CostFunctions;
using NeuralNetworks.Layers;
using NeuralNetworks.Tools;
using NeuralNetworks.Training;
using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworks.ConsoleSamples.Benchmarks
{
    public class DigitsRecognitionBenchmarks
    {
        private readonly List<TrainingElement> _trainingData;

        private readonly ITrainer _trainer;

        public DigitsRecognitionBenchmarks()
        {
            _trainingData = TrainingDataLoader.Load("NeuralNetworks.ConsoleSamples.Resources.digits-image-validation-set.json");
            
            var layers = new ILayer[] { new FullyConnectedLayer(new Sigmoid(), 30, 784), new FullyConnectedLayer(new Sigmoid(), 10, 30) };
            Randomiser.Randomise(layers, new Random(1));
            _trainer = new StochasticGradientDescent(new Quadratic(), layers, 3, 10, 1D, 0);
        }

        [Benchmark]
        public void StochasticGradientDescentBenchmark()
        {
            _trainer.Train(_trainingData.ToList());
        }
    }
}
