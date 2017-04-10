using NeuralNetworks.ActivationFunctions;
using NeuralNetworks.Extensions;
using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworks.Training
{
    public class StochasticGradientDescent : ITrainer
    {
        private readonly IActivationFunction _activationFunction;
        private readonly int _epochs;
        private readonly int _trainingBatchSize;
        private readonly double _learningRate;

        public StochasticGradientDescent(IActivationFunction activationFunction, int epochs, int trainingBatchSize, double learningRate)
        {
            _activationFunction = activationFunction;
            _epochs = epochs;
            _trainingBatchSize = trainingBatchSize;
            _learningRate = learningRate;
        }

        public void Train(NeuralNetwork neuralNetwork, List<TrainingElement> trainingData)
        {
            if (trainingData.Count < _trainingBatchSize)
            {
                throw new Exception("Training batch must be greater than trainingData size");
            }

            if (!trainingData.All(x => x.Inputs.Count == neuralNetwork.InputNeurons.Count))
            {
                throw new Exception(string.Format("All inputs of test data must be length of {0}", neuralNetwork.InputNeurons.Count));
            }

            if (!trainingData.All(x => x.ExpectedOutputs.Count == neuralNetwork.OutputNeurons.Count))
            {
                throw new Exception(string.Format("All expected outputs of test data must be length of {0}", neuralNetwork.OutputNeurons.Count));
            }

            var skip = 0;
            var epochs = _epochs;

            var neurons = neuralNetwork.HiddenNeuronLayers.Union(new List<List<Neuron>> { neuralNetwork.OutputNeurons });
            var neuronsBiases = neurons.Select(s => s.Select(x => x.Bias).ToArray()).ToArray();
            var synapsesWeights = neuralNetwork.SynapseLayers.Select(s => s.Select(x => x.Weight).ToArray()).ToArray();

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
                for (int j = 0; j < neuronsBiases[i].Length; j++)
                {
                    if (i+1 == neuronsBiases.Length)
                    {
                        neuralNetwork.OutputNeurons[j].Bias = neuronsBiases[i][j];
                    }
                    else
                    {
                        neuralNetwork.HiddenNeuronLayers[i][j].Bias = neuronsBiases[i][j];
                    }
                }
            }

            for (int i = 0; i < synapsesWeights.Length; i++)
            {
                for (int j = 0; j < synapsesWeights[i].Length; j++)
                {
                    neuralNetwork.SynapseLayers[i][j].Weight = synapsesWeights[i][j];
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
                var backpropagationResult = Backpropagation(neuronsBiases, synapsesWeights, trainingElement.Inputs, trainingElement.ExpectedOutputs);
                neuronGradients.Sum(backpropagationResult.NeuronsGradients);
                synapseGradients.Sum(backpropagationResult.SynapsesGradients);
            }


            for (int i = 0; i < neuronsBiases.Length; i++)
            {
                for (int j = 0; j < neuronsBiases[i].Length; j++)
                {
                    neuronsBiases[i][j] -= (_learningRate / _trainingBatchSize) * neuronGradients[i][j];
                }
            }

            for (int i = 0; i < synapsesWeights.Length; i++)
            {
                for (int j = 0; j < synapsesWeights[i].Length; j++)
                {
                    synapsesWeights[i][j] -= (_learningRate / _trainingBatchSize) * synapseGradients[i][j];
                }
            }
        }

        private Gradients Backpropagation(double[][] neuronsBiases, double[][] synapsesWeights, List<double> inputs, List<double> expectedOutputs)
        {
            //????
            var inputsArray = inputs.ToArray();

            var feedForwardResult = FeedForward(neuronsBiases, synapsesWeights, inputsArray);
            var neuronsInputs = feedForwardResult.ProducedInputs;
            var neuronsOutputs = feedForwardResult.ProducedOutputs;

            var neuronsGradients = neuronsBiases.CopyWithZeros();
            var synapsesGradients = synapsesWeights.CopyWithZeros();

            var outputLayerIndex = neuronsGradients.Length - 1;

            var synapseIndex = 0;

            for (int i = 0; i < neuronsGradients[outputLayerIndex].Length; i++)
            {
                var expectedOutput = expectedOutputs[i];
                var gradient = (neuronsOutputs[outputLayerIndex][i] - expectedOutput) * _activationFunction.ActivationDerivative(neuronsBiases[outputLayerIndex][i] + neuronsInputs[outputLayerIndex][i]);
                neuronsGradients[outputLayerIndex][i] = gradient;
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

                var currentLayerSynapseGrdients = synapsesGradients[i];

                synapseIndex = 0;

                for (int j = 0; j < primaryNeuronsCount; j++)
                {
                    for (int k = 0; k < targetNeuronsCount; k++)
                    {
                        layerSynapsesGradients[synapseIndex] = layerPrimaryNeuronsOutputs[j] * layerTargetNeuronsGradients[k];
                        layerPrimaryNeuronsGradients[j] += layerTargetNeuronsGradients[k] * layerSynapsesWeights[synapseIndex];
                        synapseIndex++;
                    }
                }

                for (int j = 0; j < primaryNeuronsCount; j++)
                {
                    layerPrimaryNeuronsGradients[j] *= _activationFunction.ActivationDerivative(layerPrimaryNeuronsBiases[j] + layerPrimaryNeuronsInputs[j]);
                }
            }

            var inputNeuronsCount = inputsArray.Length;
            var firstLayerNeuronsCount = neuronsInputs[0].Length;

            var firstLayerTargetNeuronsGradients = neuronsGradients[0];
            var firstLayerSynapsesGradients = synapsesGradients[0];

            synapseIndex = 0;

            for (int j = 0; j < inputNeuronsCount; j++)
            {
                for (int k = 0; k < firstLayerNeuronsCount; k++)
                {
                    firstLayerSynapsesGradients[synapseIndex] = inputsArray[j] * firstLayerTargetNeuronsGradients[k];
                    synapseIndex++;
                }
            }

            return new Gradients { NeuronsGradients = neuronsGradients, SynapsesGradients = synapsesGradients };
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
