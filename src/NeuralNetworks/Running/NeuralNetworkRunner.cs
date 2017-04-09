using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworks.Running
{
    public class NeuralNetworkRunner : IRunner
    {
        public List<double> Run(NeuralNetwork neuralNetwork, List<double> inputValues)
        {
            if (inputValues.Count != neuralNetwork.InputNeurons.Count)
            {
                throw new Exception("Number of input neurons and input values mismatch");
            }

            ClearNeuralNetwork(neuralNetwork);

            for (var index = 0; index < inputValues.Count; ++index)
            {
                neuralNetwork.InputNeurons[index].Input = inputValues[index];
            }

            for (int i = 0; i < neuralNetwork.SynapseLayers.Count; i++)
            {
                var synapseLayer = neuralNetwork.SynapseLayers[i];
                var neurons = i == 0 ? neuralNetwork.InputNeurons : neuralNetwork.HiddenNeuronLayers[i - 1];
                for (int j = 0; j < neurons.Count; j++)
                {
                    neurons[j].ProduceOutput();
                }

                for (int j = 0; j < synapseLayer.Count; j++)
                {
                    var synapse = synapseLayer[j];
                    synapse.Target.Input += synapse.Weight * synapse.Primary.Output;
                }

            }

            foreach (var outputNeuron in neuralNetwork.OutputNeurons)
            {
                outputNeuron.ProduceOutput();
            }
                
            return neuralNetwork.OutputNeurons.Select(s => s.Output).ToList();
        }

        private void ClearNeuralNetwork(NeuralNetwork neuralNetwork)
        {
            foreach (var hiddenNeuronLayer in neuralNetwork.HiddenNeuronLayers)
            {
                ClearNeuronLayer(hiddenNeuronLayer);
            }

            ClearNeuronLayer(neuralNetwork.OutputNeurons);
        }

        private void ClearNeuronLayer(List<Neuron> neurons)
        {
            foreach (var neuron in neurons)
            {
                neuron.Input = 0.0;
            }
        }
    }
}
