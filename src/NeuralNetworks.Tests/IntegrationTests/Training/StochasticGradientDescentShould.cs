using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworks.ActivationFunctions;
using NeuralNetworks.CostFunctions;
using NeuralNetworks.Layers;
using NeuralNetworks.Tools;
using NeuralNetworks.Training;
using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworks.Tests.IntegrationTests.Training
{
    [TestClass]
    public class StochasticGradientDescentShould
    {
        private ILayer[] _layers;
        private List<TrainingElement> _trainingData;

        [TestInitialize]
        public void Initialize()
        {
            _layers = new ILayer[] { new FullyConnectedLayer(new Sigmoid(), 3, 2), new FullyConnectedLayer(new Sigmoid(), 1, 3) };
            Randomiser.Randomise(_layers, new Random(5));
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
        }

        [TestMethod]
        public void ShouldTrainUsingQuadraticCostFunction()
        {
            var stochasticGradientDescent = new StochasticGradientDescent(new Quadratic(), _layers, 3000, 4, 5D, 0);
            stochasticGradientDescent.Train(_trainingData);

            var neuralNetwork = new NeuralNetwork(_layers, _layers.First().PrimaryNeuronsCount);
            var result1 = neuralNetwork.Run(_trainingData[0].Inputs);
            result1.Should().HaveCount(1);
            Math.Round(result1[0], 10).Should().Be(0.0245579310D);

            var result2 = neuralNetwork.Run(_trainingData[1].Inputs);
            result2.Should().HaveCount(1);
            Math.Round(result2[0], 10).Should().Be(0.9661695582D);

            var result3 = neuralNetwork.Run(_trainingData[2].Inputs);
            result3.Should().HaveCount(1);
            Math.Round(result3[0], 10).Should().Be(0.9852113647D);

            var result4 = neuralNetwork.Run(_trainingData[3].Inputs);
            result4.Should().HaveCount(1);
            Math.Round(result4[0], 10).Should().Be(0.0320611480D);
        }

        [TestMethod]
        public void ShouldTrainUsingCrossEntropyCostFunction()
        {
            var stochasticGradientDescent = new StochasticGradientDescent(new CrossEntropy(), _layers, 3000, 4, 5D, 0);
            stochasticGradientDescent.Train(_trainingData);

            var neuralNetwork = new NeuralNetwork(_layers, _layers.First().PrimaryNeuronsCount);
            var result1 = neuralNetwork.Run(_trainingData[0].Inputs);
            result1.Should().HaveCount(1);
            Math.Round(result1[0], 10).Should().Be(0.0005468953D);

            var result2 = neuralNetwork.Run(_trainingData[1].Inputs);
            result2.Should().HaveCount(1);
            Math.Round(result2[0], 10).Should().Be(0.9993728892D);

            var result3 = neuralNetwork.Run(_trainingData[2].Inputs);
            result3.Should().HaveCount(1);
            Math.Round(result3[0], 10).Should().Be(0.9994636693D);

            var result4 = neuralNetwork.Run(_trainingData[3].Inputs);
            result4.Should().HaveCount(1);
            Math.Round(result4[0], 10).Should().Be(0.0008765251D);
        }

        [TestMethod]
        public void ShouldTrainUsingRegularizationParam()
        {
            var stochasticGradientDescent = new StochasticGradientDescent(new CrossEntropy(), _layers, 3000, 4, 5D, 0.01D);
            stochasticGradientDescent.Train(_trainingData);

            var neuralNetwork = new NeuralNetwork(_layers, _layers.First().PrimaryNeuronsCount);
            var result1 = neuralNetwork.Run(_trainingData[0].Inputs);
            result1.Should().HaveCount(1);
            Math.Round(result1[0], 10).Should().Be(0.0285179059D);

            var result2 = neuralNetwork.Run(_trainingData[1].Inputs);
            result2.Should().HaveCount(1);
            Math.Round(result2[0], 10).Should().Be(0.9714820792D);

            var result3 = neuralNetwork.Run(_trainingData[2].Inputs);
            result3.Should().HaveCount(1);
            Math.Round(result3[0], 10).Should().Be(0.9714820797D);

            var result4 = neuralNetwork.Run(_trainingData[3].Inputs);
            result4.Should().HaveCount(1);
            Math.Round(result4[0], 10).Should().Be(0.0285163624D);
        }
    }
}
