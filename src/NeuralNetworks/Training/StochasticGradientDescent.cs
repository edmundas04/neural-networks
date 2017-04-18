using NeuralNetworks.ActivationFunctions;
using NeuralNetworks.CostFunctions;
using NeuralNetworks.Extensions;
using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworks.Training
{
    public class StochasticGradientDescent : ITrainer
    {
        private readonly IActivationFunction _activationFunction;
        private readonly ICostFunction _costFunction;
        private readonly int _epochs;
        private readonly int _trainingBatchSize;
        private readonly double _learningRate;

        public StochasticGradientDescent(IActivationFunction activationFunction, ICostFunction costFunction, int epochs, int trainingBatchSize, double learningRate)
        {
            _activationFunction = activationFunction;
            _costFunction = costFunction;
            _epochs = epochs;
            _trainingBatchSize = trainingBatchSize;
            _learningRate = learningRate;
        }

        public void Train(NeuralNetworkDto dto, List<TrainingElement> trainingData)
        {
            if (trainingData.Count < _trainingBatchSize)
            {
                throw new Exception("Training batch must be greater than trainingData size");
            }

            if (!trainingData.All(x => x.Inputs.Length == dto.InputNeuronsCount))
            {
                throw new Exception(string.Format("All inputs of test data must be length of {0}", dto.InputNeuronsCount));
            }

            if (!trainingData.All(x => x.ExpectedOutputs.Length == dto.NeuronsLayers.Last().Count))
            {
                throw new Exception(string.Format("All expected outputs of test data must be length of {0}", dto.NeuronsLayers.Last().Count));
            }

            var skip = 0;
            var epochs = _epochs;
            
            var neuronsBiases = dto.ToBiasesArray();
            var synapsesWeights = dto.ToWeightsArray();

            while (epochs-- > 0)
            {
                trainingData.Shuffle();
                while (skip + _trainingBatchSize <= trainingData.Count)
                {
                    var trainingBatch = trainingData.Skip(skip).Take(_trainingBatchSize).ToList();
                    PerformGradientDescent(neuronsBiases, synapsesWeights, trainingBatch);
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

        private void PerformGradientDescent(double[][] neuronsBiases, double[][] synapsesWeights, List<TrainingElement> trainingBatch)
        {
            var neuronGradients = neuronsBiases.CopyWithZeros();
            var synapseGradients = synapsesWeights.CopyWithZeros();

            for (int i = 0; i < trainingBatch.Count; i++)
            {
                var trainingElement = trainingBatch[i];
                var backpropagationResult = Backpropagation(synapseGradients, neuronsBiases, synapsesWeights, trainingElement.Inputs, trainingElement.ExpectedOutputs);
                neuronGradients.Sum(backpropagationResult);
            }

            for (int i = 0; i < neuronsBiases.Length; i++)
            {
                for (int j = 0; j < neuronsBiases[i].Length; j++)
                {
                    neuronsBiases[i][j] -= (_learningRate / _trainingBatchSize) * neuronGradients[i][j];
                }
            }


            var learningRate = (_learningRate / _trainingBatchSize);
            for (int i = 0; i < synapsesWeights.Length; i++)
            {
                var layerSynapsesWeights = synapsesWeights[i];
                var layerSynapsesGradients = synapseGradients[i];

                var synapsesLayerCount = layerSynapsesWeights.Length;

                for (int j = 0; j < synapsesLayerCount; j++)
                {
                    layerSynapsesWeights[j] -= learningRate * layerSynapsesGradients[j];
                }
            }
        }

        private double[][] Backpropagation(double[][] synapsesGradients, double[][] neuronsBiases, double[][] synapsesWeights, double[] inputs, double[] expectedOutputs)
        {
            var feedForwardResult = FeedForward(neuronsBiases, synapsesWeights, inputs);
            var neuronsInputs = feedForwardResult.ProducedInputs;
            var neuronsOutputs = feedForwardResult.ProducedOutputs;

            var neuronsGradients = neuronsBiases.CopyWithZeros();

            var outputLayerIndex = neuronsGradients.Length - 1;

            var synapseIndex = 0;

            for (int i = 0; i < neuronsGradients[outputLayerIndex].Length; i++)
            {
                var expectedOutput = expectedOutputs[i];
                var gradient = _costFunction.CostDerivative(neuronsOutputs[outputLayerIndex][i], expectedOutput) * _activationFunction.ActivationDerivative(neuronsBiases[outputLayerIndex][i] + neuronsInputs[outputLayerIndex][i]);
                neuronsGradients[outputLayerIndex][i] += gradient;
            }

            for (var i = synapsesGradients.Length - 1; i > 0; i--)
            {
                var layerPrimaryNeuronsOutputs = neuronsOutputs[i - 1];
                var layerPrimaryNeuronsGradients = neuronsGradients[i - 1];
                var layerPrimaryNeuronsBiases = neuronsBiases[i - 1];
                var layerPrimaryNeuronsInputs = neuronsInputs[i - 1];

                var layerTargetNeuronsGradients = neuronsGradients[i];
                var layerSynapsesGradients = synapsesGradients[i];
                var layerSynapsesWeights = synapsesWeights[i];

                var primaryNeuronsCount = neuronsInputs[i - 1].Length;
                var targetNeuronsCount = neuronsInputs[i].Length;

                synapseIndex = 0;

                for (int j = 0; j < primaryNeuronsCount; j++)
                {
                    for (int k = 0; k < targetNeuronsCount; k++)
                    {
                        layerSynapsesGradients[synapseIndex] += layerPrimaryNeuronsOutputs[j] * layerTargetNeuronsGradients[k];
                        layerPrimaryNeuronsGradients[j] += layerTargetNeuronsGradients[k] * layerSynapsesWeights[synapseIndex];
                        synapseIndex++;
                    }
                }

                for (int j = 0; j < primaryNeuronsCount; j++)
                {
                    layerPrimaryNeuronsGradients[j] *= _activationFunction.ActivationDerivative(layerPrimaryNeuronsBiases[j] + layerPrimaryNeuronsInputs[j]);
                }
            }

            var inputNeuronsCount = inputs.Length;
            var firstLayerNeuronsCount = neuronsInputs[0].Length;

            var firstLayerTargetNeuronsGradients = neuronsGradients[0];
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

            return neuronsGradients;
        }

        private FeedForwardResult FeedForward(double[][] neuronsBiases, double[][] synapsesWeights, double[] inputs)
        {
            var producedInputs = neuronsBiases.CopyWithZeros();
            var producedOutputs = neuronsBiases.CopyWithZeros();

            for (int i = 0; i < synapsesWeights.Length; i++)
            {
                var primaryNeurons = i == 0 ? inputs : producedOutputs[i - 1];
                var targetNeuronInputs = producedInputs[i];
                var targetNeuronOutputs = producedOutputs[i];
                var synapses = synapsesWeights[i];
                var primaryNeuronsCount = primaryNeurons.Length;
                var targetNeuronsCount = targetNeuronInputs.Length;
                var synapseIndex = 0;

                for (int j = 0; j < primaryNeuronsCount; j++)
                {
                    for (int k = 0; k < targetNeuronsCount; k++)
                    {
                        targetNeuronInputs[k] += primaryNeurons[j] * synapses[synapseIndex];
                        synapseIndex++;
                    }
                }

                for (int j = 0; j < targetNeuronsCount; j++)
                {
                    targetNeuronOutputs[j] = _activationFunction.Activate(targetNeuronInputs[j] + neuronsBiases[i][j]);
                }
            }

            return new FeedForwardResult { ProducedInputs = producedInputs, ProducedOutputs = producedOutputs };
        }

        private class FeedForwardResult
        {
            public double[][] ProducedInputs { get; set; }
            public double[][] ProducedOutputs { get; set; }
        }
    }
}
