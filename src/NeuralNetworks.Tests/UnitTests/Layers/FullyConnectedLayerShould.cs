using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworks.ActivationFunctions;
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
        public void ShouldProduceActivation()
        {
            _activationFunction.Activate(Arg.Is<double>(x => x == 21)).Returns(1);
            _activationFunction.Activate(Arg.Is<double>(x => x == 26)).Returns(5);
            _activationFunction.Activate(Arg.Is<double>(x => x == 31)).Returns(3);

            var fullyConnectedLayer = new FullyConnectedLayer(_activationFunction, 3, 3);
            fullyConnectedLayer.UpdateNeuronsBiases(new double[] { 1, 2, 3 });
            fullyConnectedLayer.UpdateSynapsesWeights(new double[] { 1, 2, 3, 3, 2, 1, 1, 2, 3 });

            var result = fullyConnectedLayer.ProduceActivation(new double[] { 3, 4, 5 });
            result.Should().HaveCount(3);
            result.Should().ContainInOrder(new double[] { 1, 5, 3 });
            _activationFunction.Received(3).Activate(Arg.Any<double>());

        }
    }
}
