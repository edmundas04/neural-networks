using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworks.ActivationFunctions;
using NeuralNetworks.CostFunctions;
using NeuralNetworks.Tools;
using NeuralNetworks.Training;
using System;
using System.Collections.Generic;

namespace NeuralNetworks.Tests.Integration.Training
{
    [TestClass]
    public class StochasticGradientDescentShould
    {
        private NeuralNetworkDto _logicalXORneuralNetworkDto;
        private List<TrainingElement> _trainingData;

        [TestInitialize]
        public void Initialize()
        {
            _logicalXORneuralNetworkDto = Builder.Build(new List<int> { 2, 3, 1 }, ActivationFunctionType.Sigmoid);
            Randomiser.Randomise(_logicalXORneuralNetworkDto, 5D, new Random(5));
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
            var stochasticGradientDescent = new StochasticGradientDescent(new Sigmoid(), new Quadratic(), 1000, 4, 5D);
            stochasticGradientDescent.Train(_logicalXORneuralNetworkDto, _trainingData);

            var neuralNetwork = new NeuralNetwork(_logicalXORneuralNetworkDto);
            var result1 = neuralNetwork.Run(_trainingData[0].Inputs);
            result1.Should().HaveCount(1);
            Math.Round(result1[0], 8).Should().Be(0.97302816D);

            var result2 = neuralNetwork.Run(_trainingData[1].Inputs);
            result2.Should().HaveCount(1);
            Math.Round(result2[0], 8).Should().Be(0.03993539D);

            var result3 = neuralNetwork.Run(_trainingData[2].Inputs);
            result3.Should().HaveCount(1);
            Math.Round(result3[0], 8).Should().Be(0.94297169D);

            var result4 = neuralNetwork.Run(_trainingData[3].Inputs);
            result4.Should().HaveCount(1);
            Math.Round(result4[0], 8).Should().Be(0.05742208D);
        }

        [TestMethod]
        public void ShouldTrainUsingCrossEntropyCostFunction()
        {
            var stochasticGradientDescent = new StochasticGradientDescent(new Sigmoid(), new CrossEntropy(), 1000, 4, 5D);
            stochasticGradientDescent.Train(_logicalXORneuralNetworkDto, _trainingData);

            var neuralNetwork = new NeuralNetwork(_logicalXORneuralNetworkDto);
            var result1 = neuralNetwork.Run(_trainingData[0].Inputs);
            result1.Should().HaveCount(1);
            Math.Round(result1[0], 8).Should().Be(0.99918666D);

            var result2 = neuralNetwork.Run(_trainingData[1].Inputs);
            result2.Should().HaveCount(1);
            Math.Round(result2[0], 8).Should().Be(0.00165418D);

            var result3 = neuralNetwork.Run(_trainingData[2].Inputs);
            result3.Should().HaveCount(1);
            Math.Round(result3[0], 8).Should().Be(0.99730866D);

            var result4 = neuralNetwork.Run(_trainingData[3].Inputs);
            result4.Should().HaveCount(1);
            Math.Round(result4[0], 8).Should().Be(0.00267386D);
        }
    }
}
