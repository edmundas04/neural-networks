using NeuralNetworks.ActivationFunctions;
using NeuralNetworks.Extensions;
using NeuralNetworks.Running;
using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworks.Training
{
    public class StochasticGradientDescent : ITrainer
    {
        private readonly IActivationFunction _activationFunction;
        private readonly IRunner _neuralNetworkRunner;
        private readonly int _epochs;
        private readonly int _trainingBatchSize;
        private readonly double _learningRate;

        public StochasticGradientDescent(IActivationFunction activationFunction, IRunner neuralNetworkRunner, int epochs, int trainingBatchSize, double learningRate)
        {
            _activationFunction = activationFunction;
            _neuralNetworkRunner = neuralNetworkRunner;
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
            while (epochs-- > 0)
            {
                trainingData.Shuffle();
                while (skip + _trainingBatchSize <= trainingData.Count)
                {
                    var trainingBatch = trainingData.Skip(skip).Take(_trainingBatchSize).ToList();
                    PerformGradientDescent(neuralNetwork, trainingBatch);
                    skip += _trainingBatchSize;
                }
                skip = 0;
            }
        }

        private void PerformGradientDescent(NeuralNetwork neuralNetwork, List<TrainingElement> trainingBatch)
        {
            var gradients = CreateGradients(neuralNetwork);
            var neuronGradients = gradients.NeuronsGradients;
            var synapseGradients = gradients.SynapsesGradients;

            for (int i = 0; i < trainingBatch.Count; i++)
            {
                var trainingElement = trainingBatch[i];
                var backpropagationResult = Backpropagation(neuralNetwork, trainingElement.Inputs, trainingElement.ExpectedOutputs);
                SumArrays(backpropagationResult.NeuronsGradients, neuronGradients);
                SumArrays(backpropagationResult.SynapsesGradients, synapseGradients);
            }

            for (int i = 0; i < neuralNetwork.HiddenNeuronLayers.Count; i++)
            {
                for (int j = 0; j < neuralNetwork.HiddenNeuronLayers[i].Count; j++)
                {
                    neuralNetwork.HiddenNeuronLayers[i][j].Bias -= (_learningRate / trainingBatch.Count) * neuronGradients[i][j];
                }
            }

            for (int i = 0; i < neuralNetwork.OutputNeurons.Count; i++)
            {
                neuralNetwork.OutputNeurons[i].Bias -= (_learningRate / trainingBatch.Count) * neuronGradients[neuralNetwork.HiddenNeuronLayers.Count][i];
            }

            for (int i = 0; i < neuralNetwork.SynapseLayers.Count; i++)
            {
                for (int j = 0; j < neuralNetwork.SynapseLayers[i].Count; j++)
                {
                    neuralNetwork.SynapseLayers[i][j].Weight -= (_learningRate / trainingBatch.Count) * synapseGradients[i][j];
                }
            }
        }

        private Gradients CreateGradients(NeuralNetwork neuralNetwork)
        {
            var neuronGradients = new double[neuralNetwork.HiddenNeuronLayers.Count + 1][];
            for (int i = 0; i < neuralNetwork.HiddenNeuronLayers.Count; i++)
            {
                neuronGradients[i] = new double[neuralNetwork.HiddenNeuronLayers[i].Count];
            }
            neuronGradients[neuralNetwork.HiddenNeuronLayers.Count] = new double[neuralNetwork.OutputNeurons.Count];

            var synapseGradients = new double[neuralNetwork.SynapseLayers.Count][];
            for (int i = 0; i < neuralNetwork.SynapseLayers.Count; i++)
            {
                synapseGradients[i] = new double[neuralNetwork.SynapseLayers[i].Count];
            }

            return new Gradients { SynapsesGradients = synapseGradients, NeuronsGradients = neuronGradients };
        }

        private void SumArrays(double[][] source, double[][] target)
        {
            for (int i = 0; i < source.Length; i++)
            {
                for (int j = 0; j < source[i].Length; j++)
                {
                    target[i][j] += source[i][j];
                }
            }
        }

        private Gradients Backpropagation(NeuralNetwork neuralNetwork, List<double> inputs, List<double> expectedOutputs)
        {
            _neuralNetworkRunner.Run(neuralNetwork, inputs);
            var gradients = CreateGradients(neuralNetwork);
            var neuronGradients = gradients.NeuronsGradients;
            var synapseGradients = gradients.SynapsesGradients;

            var outputNeuronGradients = neuronGradients[neuronGradients.Length - 1];

            for (int i = 0; i < outputNeuronGradients.Length; i++)
            {
                var outputNeuron = neuralNetwork.OutputNeurons[i];
                var expectedOutput = expectedOutputs[i];
                var gradient = (outputNeuron.Output - expectedOutput) * _activationFunction.ActivationDerivative(outputNeuron.Bias + outputNeuron.Input);
                outputNeuronGradients[i] = gradient;
            }

            for (var i = synapseGradients.Length - 1; i >= 0; --i)
            {
                var targetNeurons = i == synapseGradients.Length - 1 ? neuralNetwork.OutputNeurons : neuralNetwork.HiddenNeuronLayers[i];
                var primaryNeurons = i == 0 ? neuralNetwork.InputNeurons : neuralNetwork.HiddenNeuronLayers[i - 1];
                for (int j = 0; j < synapseGradients[i].Length; j++)
                {
                    var targetNeuronIndex = j % targetNeurons.Count;
                    var synapse = neuralNetwork.SynapseLayers[i][j];
                    synapseGradients[i][j] = synapse.Primary.Output * neuronGradients[i][targetNeuronIndex];

                    if(i == 0)
                    {
                        continue;
                    }

                    var primaryNeuronIndex = j % primaryNeurons.Count;
                    neuronGradients[i - 1][primaryNeuronIndex] += synapse.Weight * neuronGradients[i][targetNeuronIndex];
                }

                if (i == 0)
                {
                    continue;
                }

                for (int j = 0; j < primaryNeurons.Count; j++)
                {
                    var primaryNeuron = primaryNeurons[j];
                    neuronGradients[i - 1][j] *= _activationFunction.ActivationDerivative(primaryNeuron.Bias + primaryNeuron.Input);
                }
            }


            return gradients;
        }
    }
}
