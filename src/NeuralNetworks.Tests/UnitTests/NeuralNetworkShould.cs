﻿using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworks.Exceptions;
using NeuralNetworks.Layers;
using NSubstitute;
using System;
using System.Linq;

namespace NeuralNetworks.Tests.UnitTests
{
    [TestClass]
    public class NeuralNetworkShould
    {
        private NeuralNetworkNew _neuralNetwork;
        private ILayer _firstLayer;
        private ILayer _secondLayer;

        [TestInitialize]
        public void Initialize()
        {
            _firstLayer = Substitute.For<ILayer>();
            _secondLayer = Substitute.For<ILayer>();
            _neuralNetwork = new NeuralNetworkNew(new ILayer[] { _firstLayer, _secondLayer }, 4);
        }

        [TestMethod]
        public void ShouldThrowWhenLayersCountLessThanOne()
        {
            Action action = () => new NeuralNetworkNew(new ILayer[0], 3);
            action.ShouldThrow<NeuralNetworksException>().WithMessage("Neural network must have at least 1 layers");
        }

        [TestMethod]
        public void ShouldThrowWhenInputCountMismatch()
        {
            Action action = () => _neuralNetwork.Run(new double[5]);
            action.ShouldThrow<NeuralNetworksException>().WithMessage("Neural network has different count of inputs");
        }

        [TestMethod]
        public void ShouldRunNeuralNetwork()
        {
            var inputs = new double[] { 1, 2, 3, 4 };
            var firstLayerActivation = new double[] { 4, 5, 6, 7 };
            var secondLayerActivation = new double[] { 8, 9, 10, 11 };

            _firstLayer.ProduceActivation(Arg.Any<double[]>()).Returns(firstLayerActivation);
            _secondLayer.ProduceActivation(Arg.Any<double[]>()).Returns(secondLayerActivation);
            var result = _neuralNetwork.Run(inputs);
            _firstLayer.Received(1).ProduceActivation(Arg.Is<double[]>(x => x.SequenceEqual(inputs)));
            _secondLayer.Received(1).ProduceActivation(Arg.Is<double[]>(x => x.SequenceEqual(firstLayerActivation)));

            result.Should().NotBeNull();
            result.Should().ContainInOrder(secondLayerActivation);
        }

    }
}