using NeuralNetworks.ActivationFunctions;
using NeuralNetworks.Extensions;
using System;

namespace NeuralNetworks
{
    public class NeuralNetwork : INeuralNetwork
    {
        private readonly double[][] _neuronsBiases;
        private readonly double[][] _synapsesWeights;
        private readonly IActivationFunction _activationFunction;
        private readonly int _inputNeuronsCount;

        public NeuralNetwork(NeuralNetworkDto dto)
        {
            if(dto.NeuronsLayers.Count < 2)
            {
                throw new Exception("Neural network must have at least 2 layers");
            }

            switch (dto.ActivationFunctionType)
            {
                case ActivationFunctionType.Sigmoid:
                    _activationFunction = new Sigmoid();
                    break;
                default:
                    throw new NotSupportedException("Activation function is not suported");
            }

            _inputNeuronsCount = dto.InputNeuronsCount;

            
            _neuronsBiases = dto.ToBiasesArray();
            _synapsesWeights = dto.ToWeightsArray();
        }

        public double[] Run(double[] inputs)
        {
            var primaryNeurons = inputs;

            for (int i = 0; i < _synapsesWeights.Length; i++)
            {
                var layerSynapsesWeights = _synapsesWeights[i];
                var targetNeuronsBiases = _neuronsBiases[i];
                var targetNeurons = new double[targetNeuronsBiases.Length];
                var synapseIndex = 0;

                for (int j = 0; j < primaryNeurons.Length; j++)
                {
                    for (int k = 0; k < targetNeurons.Length; k++)
                    {
                        targetNeurons[k] += primaryNeurons[j] * layerSynapsesWeights[synapseIndex];
                        synapseIndex++;
                    }
                }

                for (int j = 0; j < targetNeurons.Length; j++)
                {
                    targetNeurons[j] = _activationFunction.Activate(targetNeurons[j] + targetNeuronsBiases[j]);
                }

                primaryNeurons = targetNeurons;
            }

            return primaryNeurons;
        }
    }
}
