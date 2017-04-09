using NeuralNetworks.ActivationFunctions;
using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworks.ConsoleSamples.Builders
{
    public class NeuralNetworkBuilder
    {
        private int _index;
        private readonly List<List<Neuron>> _neuronLayers;
        private readonly List<List<Synapse>> _synapseLayers;
        private readonly IActivationFunction _sigmoid;
        private readonly IActivationFunction _inputActivationFunction;

        public NeuralNetworkBuilder()
        {
            _neuronLayers = new List<List<Neuron>>();
            _synapseLayers = new List<List<Synapse>>();
            _sigmoid = new Sigmoid();
            _inputActivationFunction = new InputActivationFunction();
            _index = 0;
        }

        public NeuralNetworkBuilder AddNeuronLayer(int numberOfNeurons, double defaultBias, double defaultWeight)
        {
            if (numberOfNeurons < 1)
            {
                throw new ArgumentException("numberOfNeurons");
            }

            var neuronList1 = new List<Neuron>();
            var activationFunction1 = _neuronLayers.Count == 0 ? _inputActivationFunction : _sigmoid;
            while (numberOfNeurons-- > 0)
            {
                var neuronList2 = neuronList1;
                var activationFunction2 = activationFunction1;
                var bias = defaultBias;
                var index = _index;
                _index = index + 1;
                var id = index;
                var neuron = new Neuron(activationFunction2, bias, id);
                neuronList2.Add(neuron);
            }
            _neuronLayers.Add(neuronList1);

            if (_neuronLayers.Count < 2)
            {
                return this;
            }

            var list = _neuronLayers.Skip(Math.Max(0, _neuronLayers.Count() - 2)).ToList();
            ConnectLayers(list[0], list[1], defaultWeight);
            return this;
        }

        private void ConnectLayers(List<Neuron> primaryLayer, List<Neuron> targetLayer, double defaultWeight)
        {
            var synapseLayer = new List<Synapse>();
            foreach (var primaryNeuron in primaryLayer)
            {
                foreach (var targetNeuron in targetLayer)
                {
                    synapseLayer.Add(new Synapse(_index++)
                    {
                        Weight = defaultWeight,
                        Primary = primaryNeuron,
                        Target = targetNeuron
                    });
                }
            }
            _synapseLayers.Add(synapseLayer);
        }

        public NeuralNetwork Build()
        {
            _index = 0;
            if (_neuronLayers.Count < 2)
            {
                throw new Exception("Neural network is not completely built");
            }
                
            var neuralNetwork = new NeuralNetwork()
            {
                InputNeurons = _neuronLayers.First(),
                HiddenNeuronLayers = _neuronLayers.Skip(1).Take(_neuronLayers.Count - 2).ToList(),
                OutputNeurons = _neuronLayers.Last(),
                SynapseLayers = _synapseLayers.ToList()
            };
            _neuronLayers.Clear();
            _synapseLayers.Clear();
            return neuralNetwork;
        }
    }
}
