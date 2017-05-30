using NeuralNetworks.CostFunctions;
using NeuralNetworks.Exceptions;
using NeuralNetworks.Extensions;
using NeuralNetworks.Layers;
using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworks.Training
{
    public class StochasticGradientDescent : ITrainer
    {
        private readonly ICostFunction _costFunction;
        private readonly ILayer[] _layers;
        private readonly int _epochs;
        private readonly int _trainingBatchSize;
        private readonly double _learningRate;
        private readonly double _regularizationParam;
        private readonly double[][] _neuronsBiases;
        private readonly double[][] _synapsesWeights;
        private readonly double[][] _neuronsGradients;
        private readonly double[][] _synapsesGradients;
        private readonly double[][] _neuronsAdittionsToGradients;

        public StochasticGradientDescent(ICostFunction costFunction, ILayer[] layers, int epochs, int trainingBatchSize, double learningRate, double regularizationParam)
        {
            if (learningRate <= 0)
            {
                throw new ArgumentException("learningRate must be greater than zero");
            }

            if (trainingBatchSize <= 0)
            {
                throw new ArgumentException("trainingBatchSize must be greater than zero");
            }

            if (epochs <= 0)
            {
                throw new ArgumentException("epochs must be greater than zero");
            }

            if (regularizationParam < 0)
            {
                throw new ArgumentException("regularizationRate must be greater or equal to zero");
            }
            
            if(layers.Length < 2)
            {
                throw new NeuralNetworksException("At least two layers are mandatory");
            }

            _costFunction = costFunction;
            _layers = layers;
            _epochs = epochs;
            _trainingBatchSize = trainingBatchSize;
            _learningRate = learningRate;
            _regularizationParam = regularizationParam;

            _neuronsBiases = layers.Select(s => s.NeuronsBiases).ToArray();
            _synapsesWeights = layers.Select(s => s.SynapsesWeights).ToArray();
            _neuronsGradients = _neuronsBiases.CopyWithZeros();
            _synapsesGradients = _synapsesWeights.CopyWithZeros();
            _neuronsAdittionsToGradients = _neuronsBiases.CopyWithZeros();
        }

        public void Train(List<TrainingElement> trainingData)
        {
            var firstLayer = _layers.First();
            var inputNeuronsCount = firstLayer.SynapsesWeights.Length / firstLayer.NeuronsBiases.Length;
            var outputNeuronsCount = _layers.Last().NeuronsBiases.Length;

            if (trainingData.Count < _trainingBatchSize)
            {
                throw new NeuralNetworksException("Training batch must be greater than trainingData size");
            }

            if (!trainingData.All(x => x.Inputs.Length == inputNeuronsCount))
            {
                throw new NeuralNetworksException(string.Format("All inputs of test data must be length of {0}", inputNeuronsCount));
            }

            if (!trainingData.All(x => x.ExpectedOutputs.Length == outputNeuronsCount))
            {
                throw new NeuralNetworksException(string.Format("All expected outputs of test data must be length of {0}", outputNeuronsCount));
            }

            var skip = 0;
            var epochs = _epochs;
            
            while (epochs-- > 0)
            {
                trainingData.Shuffle();
                while (skip + _trainingBatchSize <= trainingData.Count)
                {
                    var trainingBatch = trainingData.Skip(skip).Take(_trainingBatchSize).ToList();
                    PerformGradientDescent(trainingBatch, trainingData.Count);
                    skip += _trainingBatchSize;
                }
                skip = 0;
            }
        }

        private void PerformGradientDescent(List<TrainingElement> trainingBatch, double trainingDataSetSize)
        {
            var neuronsBiases = _neuronsBiases;
            var synapsesWeights = _synapsesWeights;
            var neuronsGradients = _neuronsGradients;
            var synapsesGradients = _synapsesGradients;

            for (int i = 0; i < trainingBatch.Count; i++)
            {
                var trainingElement = trainingBatch[i];
                Backpropagation(trainingElement.Inputs, trainingElement.ExpectedOutputs);
            }

            var learningRateApproximation = (_learningRate / _trainingBatchSize);

            for (int i = 0; i < neuronsBiases.Length; i++)
            {
                for (int j = 0; j < neuronsBiases[i].Length; j++)
                {
                    neuronsBiases[i][j] -= learningRateApproximation * neuronsGradients[i][j];
                }
            }

            var regulatizationParamApproximation = 1 - (learningRateApproximation * (_regularizationParam / trainingDataSetSize));

            for (int i = 0; i < synapsesWeights.Length; i++)
            {
                var layerSynapsesWeights = synapsesWeights[i];
                var layerSynapsesGradients = synapsesGradients[i];

                var synapsesLayerCount = layerSynapsesWeights.Length;

                for (int j = 0; j < synapsesLayerCount; j++)
                {
                    layerSynapsesWeights[j] = (regulatizationParamApproximation * layerSynapsesWeights[j]) - (learningRateApproximation * layerSynapsesGradients[j]);
                }
            }

            neuronsGradients.FillWithZeros();
            _synapsesGradients.FillWithZeros();
        }

        private void Backpropagation(double[] inputs, double[] expectedOutputs)
        {
            var neuronsBiases = _neuronsBiases;
            var synapsesWeights = _synapsesWeights;
            var neuronsGradients = _neuronsGradients;
            var synapsesGradients = _synapsesGradients;
            var neuronsAdittionsToGradients = _neuronsAdittionsToGradients;

            var feedForwardResult = FeedForward(inputs);
            var neuronsInputs = feedForwardResult.ProducedOutputs;
            var neuronsOutputs = feedForwardResult.ProducedActivations;

            var outputLayerIndex = neuronsAdittionsToGradients.Length - 1;
            var lastLayerActivationFunction = _layers[outputLayerIndex].ActivationFunction;
            var synapseIndex = 0;

            for (int i = 0; i < neuronsAdittionsToGradients[outputLayerIndex].Length; i++)
            {
                var expectedOutput = expectedOutputs[i];
                var activationDerivative = lastLayerActivationFunction.ActivationDerivative(neuronsBiases[outputLayerIndex][i] + neuronsInputs[outputLayerIndex][i]);
                var gradient = _costFunction.CostDerivative(neuronsOutputs[outputLayerIndex][i], expectedOutput, activationDerivative);
                neuronsAdittionsToGradients[outputLayerIndex][i] += gradient;
            }

            for (var i = synapsesGradients.Length - 1; i > 0; i--)
            {
                var layerPrimaryNeuronsOutputs = neuronsOutputs[i - 1];
                var layerPrimaryNeuronsGradients = neuronsAdittionsToGradients[i - 1];
                var layerPrimaryNeuronsBiases = neuronsBiases[i - 1];
                var layerPrimaryNeuronsInputs = neuronsInputs[i - 1];

                var layerTargetNeuronsGradients = neuronsAdittionsToGradients[i];
                var layerSynapsesGradients = synapsesGradients[i];
                var layerSynapsesWeights = synapsesWeights[i];

                var primaryNeuronsCount = neuronsInputs[i - 1].Length;
                var targetNeuronsCount = neuronsInputs[i].Length;

                synapseIndex = primaryNeuronsCount * targetNeuronsCount - 1;

                for (int j = primaryNeuronsCount - 1; j >= 0; j--)
                {
                    for (int k = targetNeuronsCount - 1; k >= 0; k--)
                    {
                        var targetNeuronGradient = layerTargetNeuronsGradients[k];
                        layerSynapsesGradients[synapseIndex] += layerPrimaryNeuronsOutputs[j] * targetNeuronGradient;
                        layerPrimaryNeuronsGradients[j] += targetNeuronGradient * layerSynapsesWeights[synapseIndex];
                        synapseIndex--;
                    }
                }

                var layerActivationFunction = _layers[i].ActivationFunction;

                for (int j = 0; j < primaryNeuronsCount; j++)
                {
                    layerPrimaryNeuronsGradients[j] *= layerActivationFunction.ActivationDerivative(layerPrimaryNeuronsBiases[j] + layerPrimaryNeuronsInputs[j]);
                }
            }

            var inputNeuronsCount = inputs.Length;
            var firstLayerNeuronsCount = neuronsInputs[0].Length;

            var firstLayerTargetNeuronsGradients = neuronsAdittionsToGradients[0];
            var firstLayerSynapsesGradients = synapsesGradients[0];

            synapseIndex = 0;

            for (int j = 0; j < inputNeuronsCount; j++)
            {
                for (int k = 0; k < firstLayerNeuronsCount; k++)
                {
                    firstLayerSynapsesGradients[synapseIndex] += inputs[j] * firstLayerTargetNeuronsGradients[k];
                    synapseIndex++;
                }
            }

            neuronsGradients.Sum(neuronsAdittionsToGradients);
            _neuronsAdittionsToGradients.FillWithZeros();
        }

        private FeedForwardResult FeedForward(double[] inputs)
        {
            var layersCount = _layers.Length;

            var producedOutputs = new double[layersCount][];
            var producedActivations = new double[layersCount][];

            var perviousLayerActivations = inputs;
            for (int i = 0; i < layersCount; i++)
            {
                var layer = _layers[i];
                layer.Produce(perviousLayerActivations);
                producedOutputs[i] = layer.Outputs;
                var activations = layer.Activations;
                producedActivations[i] = activations;
                perviousLayerActivations = activations;
            }

            return new FeedForwardResult { ProducedOutputs = producedOutputs, ProducedActivations = producedActivations };
        }

        private class FeedForwardResult
        {
            public double[][] ProducedOutputs { get; set; }
            public double[][] ProducedActivations { get; set; }
        }
    }
}
