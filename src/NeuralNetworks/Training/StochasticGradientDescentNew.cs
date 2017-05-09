using NeuralNetworks.ActivationFunctions;
using NeuralNetworks.CostFunctions;
using NeuralNetworks.Exceptions;
using NeuralNetworks.Extensions;
using NeuralNetworks.Layers;
using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworks.Training
{
    public class StochasticGradientDescentNew : ITrainer
    {
        private readonly ICostFunction _costFunction;
        private readonly int _epochs;
        private readonly int _trainingBatchSize;
        private readonly double _learningRate;
        private readonly double _regularizationParam;

        public StochasticGradientDescentNew(ICostFunction costFunction, int epochs, int trainingBatchSize, double learningRate, double regularizationParam)
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
            
            _costFunction = costFunction;
            _epochs = epochs;
            _trainingBatchSize = trainingBatchSize;
            _learningRate = learningRate;
            _regularizationParam = regularizationParam;
        }

        //TODO: Remove this code after refactoring
        private ILayer[] ToLayers(NeuralNetworkDto neuralNetworkDto)
        {
            var result = new ILayer[neuralNetworkDto.NeuronsLayers.Count];
            var primaryNeuronsCount = neuralNetworkDto.InputNeuronsCount;

            for (int i = 0; i < neuralNetworkDto.NeuronsLayers.Count; i++)
            {
                var neurons = neuralNetworkDto.NeuronsLayers[i];
                var synapses = neuralNetworkDto.SynapsesLayers[i];

                var layer = new FullyConnectedLayer(new Sigmoid(), synapses.Select(s => s.Weight).ToArray(), neurons.Select(s => s.Bias).ToArray());
                result[i] = layer;
                primaryNeuronsCount = neurons.Count;
            }

            return result;
        }

        public void Train(NeuralNetworkDto dto, List<TrainingElement> trainingData)
        {
            var layers = ToLayers(dto);
            var firstLayer = layers.First();
            var inputNeuronsCount = firstLayer.SynapsesWeights.Length / firstLayer.NeuronsBiases.Length;
            var outputNeuronsCount = layers.Last().NeuronsBiases.Length;

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

            var neuronsBiases = layers.Select(s => s.NeuronsBiases).ToArray();
            var synapsesWeights = layers.Select(s => s.SynapsesWeights).ToArray();

            while (epochs-- > 0)
            {
                trainingData.Shuffle();
                while (skip + _trainingBatchSize <= trainingData.Count)
                {
                    var trainingBatch = trainingData.Skip(skip).Take(_trainingBatchSize).ToList();
                    PerformGradientDescent(layers, trainingBatch, neuronsBiases, synapsesWeights, trainingData.Count);
                    skip += _trainingBatchSize;
                }
                skip = 0;
            }

            for (int i = 0; i < neuronsBiases.Length; i++)
            {
                var positionNeuronMap = dto.NeuronsLayers[i].ToDictionary(x => x.Position);

                for (int j = 0; j < neuronsBiases[i].Length; j++)
                {
                    positionNeuronMap[j].Bias = neuronsBiases[i][j];
                }
            }

            for (int i = 0; i < synapsesWeights.Length; i++)
            {
                var targetNeuronsCount = neuronsBiases[i].Length;
                var positionSynapseMap = dto.SynapsesLayers[i].ToDictionary(x => x.PrimaryNeuronPosition * targetNeuronsCount + x.TargetNeuronPosition);

                for (int j = 0; j < synapsesWeights[i].Length; j++)
                {
                    positionSynapseMap[j].Weight = synapsesWeights[i][j];
                }

            }
        }

        private void PerformGradientDescent(ILayer[] layers, List<TrainingElement> trainingBatch, double[][] neuronsBiases, double[][] synapsesWeights, double trainingDataSetSize)
        {
            var neuronGradients = neuronsBiases.CopyWithZeros();
            var synapseGradients = synapsesWeights.CopyWithZeros();

            for (int i = 0; i < trainingBatch.Count; i++)
            {
                var trainingElement = trainingBatch[i];
                Backpropagation(layers, synapseGradients, neuronGradients, neuronsBiases, synapsesWeights, trainingElement.Inputs, trainingElement.ExpectedOutputs);
            }

            var learningRateApproximation = (_learningRate / _trainingBatchSize);

            for (int i = 0; i < neuronsBiases.Length; i++)
            {
                for (int j = 0; j < neuronsBiases[i].Length; j++)
                {
                    neuronsBiases[i][j] -= learningRateApproximation * neuronGradients[i][j];
                }
            }

            var regulatizationParamApproximation = 1 - (learningRateApproximation * (_regularizationParam / trainingDataSetSize));

            for (int i = 0; i < synapsesWeights.Length; i++)
            {
                var layerSynapsesWeights = synapsesWeights[i];
                var layerSynapsesGradients = synapseGradients[i];

                var synapsesLayerCount = layerSynapsesWeights.Length;

                for (int j = 0; j < synapsesLayerCount; j++)
                {
                    layerSynapsesWeights[j] = (regulatizationParamApproximation * layerSynapsesWeights[j]) - (learningRateApproximation * layerSynapsesGradients[j]);
                }
            }
        }

        private void Backpropagation(ILayer[] layers, double[][] synapsesGradients, double[][] neuronsGradients, double[][] neuronsBiases, double[][] synapsesWeights, double[] inputs, double[] expectedOutputs)
        {
            var feedForwardResult = FeedForward(layers, inputs);
            var neuronsInputs = feedForwardResult.ProducedOutputs;
            var neuronsOutputs = feedForwardResult.ProducedActivations;

            var neuronsAdittionsToGradients = neuronsBiases.CopyWithZeros();
            var outputLayerIndex = neuronsAdittionsToGradients.Length - 1;
            var lastLayerActivationFunction = layers[outputLayerIndex].ActivationFunction;
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

                var layerActivationFunction = layers[i].ActivationFunction;

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
        }

        private FeedForwardResult FeedForward(ILayer[] layers, double[] inputs)
        {
            var layersCount = layers.Length;

            var producedOutputs = new double[layersCount][];
            var producedActivations = new double[layersCount][];

            var perviousLayerActivations = inputs;
            for (int i = 0; i < layersCount; i++)
            {
                var layer = layers[i];
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
