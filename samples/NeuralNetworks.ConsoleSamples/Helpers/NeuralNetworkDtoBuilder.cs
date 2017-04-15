using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworks.ConsoleSamples.Helpers
{
    public static class NeuralNetworkDtoBuilder
    {
        public static NeuralNetworkDto Build(List<int> neuronsInLayerCounts, ActivationFunctionType activationFunctionType)
        {
            if (neuronsInLayerCounts.Count < 3)
            {
                throw new Exception("Neural network must contain at least 3 layers of neurons");
            }

            if (!neuronsInLayerCounts.All(x => x > 0))
            {
                throw new Exception("All layers must have at least 1 neuron");
            }

            var result = new NeuralNetworkDto
            {
                InputNeuronsCount = neuronsInLayerCounts.First(),
                ActivationFunctionType = activationFunctionType,
                SynapsesLayers = new List<List<SynapseDto>>(),
                NeuronsLayers = new List<List<NeuronDto>>()
            };

            for (int i = 1; i < neuronsInLayerCounts.Count; i++)
            {
                var neuronsInLayerCount = neuronsInLayerCounts[i];
                var neurons = new List<NeuronDto>();

                for (int j = 0; j < neuronsInLayerCount; j++)
                {
                    neurons.Add(new NeuronDto { Position = j });
                }

                result.NeuronsLayers.Add(neurons);
                result.SynapsesLayers.Add(ConnectNeurons(neuronsInLayerCounts[i - 1], neuronsInLayerCounts[i]));
            }

            return result;
        }

        private static List<SynapseDto> ConnectNeurons(int primaryLayerCount, int targetLayerCount)
        {
            var result = new List<SynapseDto>();

            for (int i = 0; i < primaryLayerCount; i++)
            {
                for (int j = 0; j < targetLayerCount; j++)
                {
                    result.Add(new SynapseDto { PrimaryNeuronPosition = i, TargetNeuronPosition = j });
                }
            }

            return result;
        }
    }
}
