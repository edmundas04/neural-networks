using System.Collections.Generic;

namespace NeuralNetworks
{
    public class NeuralNetwork
    {
        public List<Neuron> InputNeurons { get; set; }

        public List<List<Neuron>> HiddenNeuronLayers { get; set; }

        public List<Neuron> OutputNeurons { get; set; }

        public List<List<Synapse>> SynapseLayers { get; set; }
    }
}
