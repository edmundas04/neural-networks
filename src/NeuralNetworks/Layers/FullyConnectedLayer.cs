using NeuralNetworks.ActivationFunctions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks.Layers
{
    public class FullyConnectedLayer : ILayer
    {
        private readonly IActivationFunction _activationFunction;
        private readonly double[] _synapsesWeights;
        private readonly double[] _neuronsBiases;
        private readonly int _primaryNeuronsCount;

        public FullyConnectedLayer(IActivationFunction activationFunction, int currentNeuronsCount, int primaryNeuronsCount)
        {
            if(currentNeuronsCount < 1)
            {
                throw new ArgumentException("neuronsCount must be greater than zero");
            }

            if(primaryNeuronsCount < 1)
            {
                throw new ArgumentException("primaryNeuronsCount must be greater than zero");
            }

            _activationFunction = activationFunction ?? throw new ArgumentException("activationFunction is null");
            _synapsesWeights = new double[currentNeuronsCount * primaryNeuronsCount];
            _neuronsBiases = new double[currentNeuronsCount];
            _primaryNeuronsCount = primaryNeuronsCount;
        }

        public void UpdateSynapsesWeights(double[] newSynapsesWeighs)
        {
            var newSynapsesWeighsCount = newSynapsesWeighs.Length;

            for (int i = 0; i < newSynapsesWeighsCount; i++)
            {
                _synapsesWeights[i] = newSynapsesWeighs[i];
            }
        }

        public void UpdateNeuronsBiases(double[] newNeuronsBiases)
        {
            var newNeuronsBiasesCount = _neuronsBiases.Length;

            for (int i = 0; i < newNeuronsBiasesCount; i++)
            {
                _neuronsBiases[i] = newNeuronsBiases[i];
            }
        }

        public double[] ProduceActivation(double[] input)
        {
            var currentNeuronsCount = _neuronsBiases.Length;

            var result = new double[currentNeuronsCount];

            var synapseIndex = 0;

            for (int i = 0; i < _primaryNeuronsCount; i++)
            {
                for (int j = 0; j < currentNeuronsCount; j++)
                {
                    result[j] += _synapsesWeights[synapseIndex] * input[i];
                    synapseIndex++;
                }
            }

            for (int i = 0; i < currentNeuronsCount; i++)
            {
                result[i] = _activationFunction.Activate(result[i] + _neuronsBiases[i]);
            }

            return result;
        }


    }
}
