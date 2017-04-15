using System;
using System.Collections.Generic;

namespace NeuralNetworks.Extensions
{
    internal static class NeuralNetworkDtoExtensions
    {
        internal static double[][] ToBiasesArray(this NeuralNetworkDto dto)
        {
            var usedPositions = new HashSet<int>();
            var neuronsBiases = new double[dto.NeuronsLayers.Count][];

            for (int i = 0; i < dto.NeuronsLayers.Count; i++)
            {
                neuronsBiases[i] = new double[dto.NeuronsLayers[i].Count];

                for (int j = 0; j < dto.NeuronsLayers[i].Count; j++)
                {
                    var neuron = dto.NeuronsLayers[i][j];

                    if (usedPositions.Contains(neuron.Position))
                    {
                        throw new Exception("Layer has neurons with duplicated positions");
                    }
                    usedPositions.Add(neuron.Position);
                    neuronsBiases[i][neuron.Position] = neuron.Bias;
                }

                usedPositions.Clear();
            }

            return neuronsBiases;
        }

        internal static double[][] ToWeightsArray(this NeuralNetworkDto dto)
        {
            var synapsesWeights = new double[dto.SynapsesLayers.Count][];
            var usedPositions = new HashSet<int>();

            for (int i = 0; i < dto.SynapsesLayers.Count; i++)
            {
                var targetNeuronLayerCount = dto.NeuronsLayers[i].Count;
                synapsesWeights[i] = new double[dto.SynapsesLayers[i].Count];

                for (int j = 0; j < dto.SynapsesLayers[i].Count; j++)
                {
                    var synapse = dto.SynapsesLayers[i][j];
                    var synapseIndex = synapse.PrimaryNeuronPosition * targetNeuronLayerCount + synapse.TargetNeuronPosition;

                    if (usedPositions.Contains(synapseIndex))
                    {
                        throw new Exception("Layer has synapses with duplicated positions");
                    }
                    usedPositions.Add(synapseIndex);
                    synapsesWeights[i][synapseIndex] = synapse.Weight;
                }

                usedPositions.Clear();
            }

            return synapsesWeights;
        }
    }
}
