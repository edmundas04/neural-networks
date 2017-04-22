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
            result1[0].Should().Be(0.97302815819753175D);

            var result2 = neuralNetwork.Run(_trainingData[1].Inputs);
            result2.Should().HaveCount(1);
            result2[0].Should().Be(0.039935388558418453D);

            var result3 = neuralNetwork.Run(_trainingData[2].Inputs);
            result3.Should().HaveCount(1);
            result3[0].Should().Be(0.94297168622448257D);

            var result4 = neuralNetwork.Run(_trainingData[3].Inputs);
            result4.Should().HaveCount(1);
            result4[0].Should().Be(0.057422082340008042D);
        }

        [TestMethod]
        public void ShouldTrainUsingCrossEntropyCostFunction()
        {
            var stochasticGradientDescent = new StochasticGradientDescent(new Sigmoid(), new CrossEntropy(), 1000, 4, 5D);
            stochasticGradientDescent.Train(_logicalXORneuralNetworkDto, _trainingData);

            var neuralNetwork = new NeuralNetwork(_logicalXORneuralNetworkDto);
            var result1 = neuralNetwork.Run(_trainingData[0].Inputs);
            result1.Should().HaveCount(1);
            result1[0].Should().Be(0.999186664461933D);

            var result2 = neuralNetwork.Run(_trainingData[1].Inputs);
            result2.Should().HaveCount(1);
            result2[0].Should().Be(0.0016541769597805038D);

            var result3 = neuralNetwork.Run(_trainingData[2].Inputs);
            result3.Should().HaveCount(1);
            result3[0].Should().Be(0.99730866387705452D);

            var result4 = neuralNetwork.Run(_trainingData[3].Inputs);
            result4.Should().HaveCount(1);
            result4[0].Should().Be(0.0026738630799789296D);
        }
    }
}
