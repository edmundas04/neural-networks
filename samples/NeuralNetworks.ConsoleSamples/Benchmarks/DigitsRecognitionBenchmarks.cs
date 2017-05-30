using BenchmarkDotNet.Attributes;
using NeuralNetworks.ActivationFunctions;
using NeuralNetworks.ConsoleSamples.Extensions;
using NeuralNetworks.ConsoleSamples.Helpers;
using NeuralNetworks.CostFunctions;
using NeuralNetworks.Tools;
using NeuralNetworks.Training;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks.ConsoleSamples.Benchmarks
{
    public class DigitsRecognitionBenchmarks
    {
        private readonly List<TrainingElement> _trainingData;
        private readonly ITrainer _trainer1;
        private readonly NeuralNetworkDto _neuralNetworkDto1;

        private readonly ITrainer _trainer2;
        private readonly NeuralNetworkDto _neuralNetworkDto2;

        public DigitsRecognitionBenchmarks()
        {
            _trainingData = TrainingDataLoader.Load("NeuralNetworks.ConsoleSamples.Resources.digits-image-validation-set.json");

            _neuralNetworkDto1 = Builder.Build(new List<int> { 784, 30, 10 }, ActivationFunctionType.Sigmoid);
            Randomiser.Randomise(_neuralNetworkDto1, new Random(1));
            _trainer1 = new StochasticGradientDescent(new Sigmoid(), new Quadratic(), 3, 10, 1D, 0);

            _neuralNetworkDto2 = Builder.Build(new List<int> { 784, 30, 10 }, ActivationFunctionType.Sigmoid);
            Randomiser.Randomise(_neuralNetworkDto2, new Random(1));
            _trainer2 = new StochasticGradientDescentNew(new Quadratic(), _neuralNetworkDto2.ToLayers(), 3, 10, 1D, 0);
        }

        [Benchmark]
        public void StochasticGradientDescentBenchmark()
        {
            _trainer1.Train(_neuralNetworkDto1, _trainingData.ToList());
        }

        [Benchmark]
        public void StochasticGradientDescentNewBenchmark()
        {
            _trainer2.Train(_neuralNetworkDto2, _trainingData.ToList());
        }
    }
}
