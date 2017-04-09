using NeuralNetworks.ActivationFunctions;

namespace NeuralNetworks
{
    public class Neuron
    {
        private readonly IActivationFunction _activationFunction;

        public int Id { get; }

        public double Bias { get; set; }

        public double Input { get; set; }

        public double Output { get; private set; }

        public Neuron(IActivationFunction activationFunction, double bias, int id)
        {
            Id = id;
            Bias = bias;
            _activationFunction = activationFunction;
        }

        public void ProduceOutput()
        {
            Output = _activationFunction.Activate(Input + Bias);
        }
    }
}
