using System;

namespace NeuralNetworks.ActivationFunctions
{
    public class InputActivationFunction : IActivationFunction
    {
        public double Activate(double z)
        {
            return z;
        }

        double IActivationFunction.ActivationDerivative(double z)
        {
            throw new NotSupportedException("Derivative of this activation function is irrelevant");
        }
    }
}
