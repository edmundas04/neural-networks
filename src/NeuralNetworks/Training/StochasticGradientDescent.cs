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
                
            var count = 0;
            var epochs = _epochs;
            while (epochs-- > 0)
            {
                trainingData.Shuffle();
                while (count + _trainingBatchSize <= trainingData.Count)
                {
                    var trainingBatch = trainingData.Skip(count).Take(_trainingBatchSize).ToList();
                    PerformGradientDescent(neuralNetwork, trainingBatch);
                    count += _trainingBatchSize;
                }
                count = 0;
            }
        }

        private void PerformGradientDescent(NeuralNetwork neuralNetwork, List<TrainingElement> trainingBatch)
        {
            var neurons = neuralNetwork.HiddenNeuronLayers.SelectMany(s => s).Union(neuralNetwork.OutputNeurons).ToList();
            var synapses = neuralNetwork.SynapseLayers.SelectMany(s => s).ToList();

            var neuronGradientMap = neurons.ToDictionary(x => x.Id, x => 0.0);
            var synapseGradientMap = synapses.ToDictionary(x => x.Id, x => 0.0);

            foreach (var trainingElement in trainingBatch)
            {
                var backpropagationResult = Backpropagation(neuralNetwork, trainingElement.Inputs, trainingElement.ExpectedOutputs);
                foreach (var biasesGradient in backpropagationResult.BiasesGradients)
                {
                    var id = biasesGradient.Id;
                    neuronGradientMap[id] = neuronGradientMap[id] + biasesGradient.Gradient;
                }
                foreach (var weightsGradient in backpropagationResult.WeightsGradients)
                {
                    var id = weightsGradient.Id;
                    synapseGradientMap[id] = synapseGradientMap[id] + weightsGradient.Gradient;
                }
            }

            foreach (var neuron in neurons)
            {
                neuron.Bias -= _learningRate / trainingBatch.Count * neuronGradientMap[neuron.Id];
            }
                
            foreach (var synapse in synapses)
            {
                synapse.Weight -= _learningRate / trainingBatch.Count * synapseGradientMap[synapse.Id];
            }
        }

        private BackpropagationResult Backpropagation(NeuralNetwork neuralNetwork, List<double> inputs, List<double> expectedOutputs)
        {
            _neuralNetworkRunner.Run(neuralNetwork, inputs);
            var neuronGradientMap = new Dictionary<int, double>();
            for (var index = 0; index < neuralNetwork.OutputNeurons.Count; ++index)
            {
                var outputNeuron = neuralNetwork.OutputNeurons[index];
                var expectedOutput = expectedOutputs[index];
                var num = (outputNeuron.Output - expectedOutput) * _activationFunction.ActivationDerivative(outputNeuron.Bias + outputNeuron.Input);
                neuronGradientMap.Add(outputNeuron.Id, num);
            }

            for (var i = neuralNetwork.SynapseLayers.Count - 1; i >= 0; --i)
            {
                var synapseLayer = neuralNetwork.SynapseLayers[i];
                var intermediateNeuronGradientMap = new Dictionary<int, double>();
                var neurons = new List<Neuron>();
                for (var j = 0; j < synapseLayer.Count; ++j)
                {
                    var synapse = synapseLayer[j];
                    if (!intermediateNeuronGradientMap.ContainsKey(synapse.Primary.Id))
                    {
                        intermediateNeuronGradientMap.Add(synapse.Primary.Id, 0.0);
                        neurons.Add(synapse.Primary);
                    }
                    var dictionary2 = intermediateNeuronGradientMap;
                    var id = synapse.Primary.Id;
                    dictionary2[id] = dictionary2[id] + synapse.Weight * neuronGradientMap[synapse.Target.Id];
                }

                foreach (var neuron in neurons)
                {
                    neuronGradientMap.Add(neuron.Id, intermediateNeuronGradientMap[neuron.Id] * _activationFunction.ActivationDerivative(neuron.Bias + neuron.Input));
                }
                    
            }

            var weightsGradients = new List<BackpropagationItem>();
            foreach (var synapseLayer in neuralNetwork.SynapseLayers)
            {
                foreach (var synapse in synapseLayer)
                {
                    var backpropagationItem = new BackpropagationItem();
                    backpropagationItem.Id = synapse.Id;
                    var num = neuronGradientMap[synapse.Target.Id];
                    backpropagationItem.Gradient = synapse.Primary.Output * num;
                    weightsGradients.Add(backpropagationItem);
                }
            }
            var biasesGradients = neuralNetwork.HiddenNeuronLayers.SelectMany(s => s).Union(neuralNetwork.OutputNeurons).Select(s => new BackpropagationItem()
            {
                Id = s.Id,
                Gradient = neuronGradientMap[s.Id]
            }).ToList();

            return new BackpropagationResult()
            {
                BiasesGradients = biasesGradients,
                WeightsGradients = weightsGradients
            };
        }
    }
}
