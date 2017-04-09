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
                neuralNetwork.InputNeurons[index].ProduceOutput();
            }
            var intSet = new HashSet<int>();
            foreach (var synapseLayer in neuralNetwork.SynapseLayers)
            {
                foreach (var synapse in synapseLayer)
                {
                    if (!intSet.Contains(synapse.Primary.Id))
                    {
                        intSet.Add(synapse.Primary.Id);
                        synapse.Primary.ProduceOutput();
                    }
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
