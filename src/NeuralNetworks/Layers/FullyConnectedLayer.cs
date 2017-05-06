using NeuralNetworks.ActivationFunctions;
using NeuralNetworks.Exceptions;
using System;

namespace NeuralNetworks.Layers
{
    public class FullyConnectedLayer : ILayer
    {
        public IActivationFunction ActivationFunction { get; }
        public double[] Outputs { get; }
        public double[] Activations { get; }
        public double[] SynapsesWeights { get; }
        public double[] NeuronsBiases { get; }
        private readonly int _primaryNeuronsCount;

        public FullyConnectedLayer(IActivationFunction activationFunction, int currentNeuronsCount, int primaryNeuronsCount)
        {
            if (currentNeuronsCount < 1)
            {
                throw new ArgumentException("neuronsCount must be greater than zero");
            }

            if (primaryNeuronsCount < 1)
            {
                throw new ArgumentException("primaryNeuronsCount must be greater than zero");
            }

            ActivationFunction = activationFunction ?? throw new ArgumentException("activationFunction is null");
            Outputs = new double[currentNeuronsCount];
            Activations = new double[currentNeuronsCount];
            SynapsesWeights = new double[currentNeuronsCount * primaryNeuronsCount];
            NeuronsBiases = new double[currentNeuronsCount];
            _primaryNeuronsCount = primaryNeuronsCount;
        }

        public FullyConnectedLayer(IActivationFunction activationFunction, double[] synapsesWeights, double[] neuronsBiases)
        {
            if (neuronsBiases.Length < 1)
            {
                throw new NeuralNetworksException("Amount of neurons must be greater than zero");
            }

            _primaryNeuronsCount = synapsesWeights.Length / neuronsBiases.Length;

            if (_primaryNeuronsCount < 1)
            {
                throw new NeuralNetworksException("Amount of primary neurons must be greater than zero");
            }

            if (synapsesWeights.Length % neuronsBiases.Length != 0)
            {
                throw new NeuralNetworksException("Incorrect number of weights");
            }

            
            ActivationFunction = activationFunction ?? throw new ArgumentException("activationFunction is null");
            Outputs = new double[neuronsBiases.Length];
            Activations = new double[neuronsBiases.Length];
            SynapsesWeights = synapsesWeights;
            NeuronsBiases = neuronsBiases;
        }

        public void Produce(double[] input)
        {
            var neuronsBiases = NeuronsBiases;
            var synapsesWeights = SynapsesWeights;

            var currentNeuronsCount = neuronsBiases.Length;
            var activationFunction = ActivationFunction;
            var outputs = Outputs;
            Array.Clear(outputs, 0, currentNeuronsCount);
            var activations = Activations;            

            var synapseIndex = 0;

            for (int i = 0; i < _primaryNeuronsCount; i++)
            {
                for (int j = 0; j < currentNeuronsCount; j++)
                {
                    outputs[j] += synapsesWeights[synapseIndex] * input[i];
                    synapseIndex++;
                }
            }

            for (int i = 0; i < currentNeuronsCount; i++)
            {
                activations[i] = activationFunction.Activate(outputs[i] + neuronsBiases[i]);
            }
        }
    }
}
