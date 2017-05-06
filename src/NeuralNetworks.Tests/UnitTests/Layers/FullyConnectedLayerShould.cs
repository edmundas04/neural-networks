using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworks.ActivationFunctions;
using NeuralNetworks.Exceptions;
using NeuralNetworks.Layers;
using NSubstitute;
using System;

namespace NeuralNetworks.Tests.UnitTests.Layers
{
    [TestClass]
    public class FullyConnectedLayerShould
    {
        private IActivationFunction _activationFunction;

        [TestInitialize]
        public void Initialize()
        {
            _activationFunction = Substitute.For<IActivationFunction>();
        }

        [TestMethod]
        public void ShouldThrowWhenActivationIsNull()
        {
            Action action= () => new FullyConnectedLayer(null, 1, 1);
            action.ShouldThrow<ArgumentException>().WithMessage("activationFunction is null");
        }

        [TestMethod]
        public void ShouldThrowWhenNeuronsCountLessThanOne()
        {
            Action action = () => new FullyConnectedLayer(_activationFunction, -6, 1);
            action.ShouldThrow<ArgumentException>().WithMessage("neuronsCount must be greater than zero");
        }

        [TestMethod]
        public void ShouldThrowWhenPrimaryNeuronsCountLessThanOne()
        {
            Action action = () => new FullyConnectedLayer(_activationFunction, 1, -6);
            action.ShouldThrow<ArgumentException>().WithMessage("primaryNeuronsCount must be greater than zero");
        }

        [TestMethod]
        public void ShouldThrowWhenIncorrectNumberOfWeights()
        {
            Action action = () => new FullyConnectedLayer(_activationFunction, new double[5], new double[2]);
            action.ShouldThrow<NeuralNetworksException>().WithMessage("Incorrect number of weights");
        }

        [TestMethod]
        public void ShouldThrowWhenNeuronsArrayIsEmpty()
        {
            Action action = () => new FullyConnectedLayer(_activationFunction, new double[5], new double[0]);
            action.ShouldThrow<NeuralNetworksException>().WithMessage("Amount of neurons must be greater than zero");
        }

        [TestMethod]
        public void ShouldThrowWhenSynapsesArrayIsEmpty()
        {
            Action action = () => new FullyConnectedLayer(_activationFunction, new double[0], new double[5]);
            action.ShouldThrow<NeuralNetworksException>().WithMessage("Amount of primary neurons must be greater than zero");
        }

        [TestMethod]
        public void ShouldInitializeWhenCountsProvided()
        {
            var fullyConnectedLayer = new FullyConnectedLayer(_activationFunction, 8, 4);
            fullyConnectedLayer.ActivationFunction.Should().NotBeNull();
            fullyConnectedLayer.SynapsesWeights.Should().NotBeNull();
            fullyConnectedLayer.SynapsesWeights.Should().ContainInOrder(new double[] { 0, 0, 0, 0, 0, 0, 0, 0 });
            fullyConnectedLayer.NeuronsBiases.Should().NotBeNull();
            fullyConnectedLayer.NeuronsBiases.Should().ContainInOrder(new double[] { 0, 0, 0, 0 });
            fullyConnectedLayer.Outputs.Should().NotBeNull();
            fullyConnectedLayer.Outputs.Should().ContainInOrder(new double[] { 0, 0, 0, 0 });
            fullyConnectedLayer.Activations.Should().NotBeNull();
            fullyConnectedLayer.Activations.Should().ContainInOrder(new double[] { 0, 0, 0, 0 });
        }

        [TestMethod]
        public void ShouldInitializeWhenValuesProvided()
        {
            var fullyConnectedLayer = new FullyConnectedLayer(_activationFunction, new double[] { 1, 2, 3, 3, 2, 1, 1, 2, 3 }, new double[] { 1, 2, 3 });
            fullyConnectedLayer.ActivationFunction.Should().NotBeNull();
            fullyConnectedLayer.SynapsesWeights.Should().NotBeNull();
            fullyConnectedLayer.SynapsesWeights.Should().ContainInOrder(new double[] { 1, 2, 3, 3, 2, 1, 1, 2, 3 });
            fullyConnectedLayer.NeuronsBiases.Should().NotBeNull();
            fullyConnectedLayer.NeuronsBiases.Should().ContainInOrder(new double[] { 1, 2, 3 });
            fullyConnectedLayer.Outputs.Should().NotBeNull();
            fullyConnectedLayer.Outputs.Should().ContainInOrder(new double[] { 0, 0, 0 });
            fullyConnectedLayer.Activations.Should().NotBeNull();
            fullyConnectedLayer.Activations.Should().ContainInOrder(new double[] { 0, 0, 0 });
        }

        [TestMethod]
        public void ShouldProduceActivation()
        {
            _activationFunction.Activate(Arg.Is<double>(x => x == 21)).Returns(1);
            _activationFunction.Activate(Arg.Is<double>(x => x == 26)).Returns(5);
            _activationFunction.Activate(Arg.Is<double>(x => x == 31)).Returns(3);

            var fullyConnectedLayer = new FullyConnectedLayer(_activationFunction, new double[] { 1, 2, 3, 3, 2, 1, 1, 2, 3 }, new double[] { 1, 2, 3 });

            fullyConnectedLayer.Produce(new double[] { 3, 4, 5 });
            fullyConnectedLayer.Activations.Should().NotBeNull();
            fullyConnectedLayer.Activations.Should().ContainInOrder(new double[] { 1, 5, 3 });
            _activationFunction.Received(3).Activate(Arg.Any<double>());

        }
    }
}
