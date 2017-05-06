using NeuralNetworks.Exceptions;
using NeuralNetworks.Layers;

namespace NeuralNetworks
{
    public class NeuralNetwork : INeuralNetwork
    {
        private readonly ILayer[] _layers;
        private readonly int _inputNeuronsCount;

        public NeuralNetwork(ILayer[] layers, int inputNeuronsCount)
        {
            if(layers.Length < 1)
            {
                throw new NeuralNetworksException("Neural network must have at least 1 layers");
            }

            _layers = layers;
            _inputNeuronsCount = inputNeuronsCount;
        }

        public double[] Run(double[] inputs)
        {
            if(_inputNeuronsCount != inputs.Length)
            {
                throw new NeuralNetworksException("Neural network has different count of inputs");
            }

            var activations = inputs;
            var layersCount = _layers.Length;

            for (int i = 0; i < layersCount; i++)
            {
                var layer = _layers[i];
                layer.Produce(activations);
                activations = layer.Activations;
            }

            return activations;
        }
    }
}
