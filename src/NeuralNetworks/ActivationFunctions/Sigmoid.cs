using System;

namespace NeuralNetworks.ActivationFunctions
{
    public class Sigmoid : IActivationFunction
    {
        public double Activate(double z)
        {
            return 1.0 / (1.0 + Math.Exp(-z));
        }

        public double ActivationDerivative(double z)
        {
            return Activate(z) * (1.0 - Activate(z));
        }
    }
}
