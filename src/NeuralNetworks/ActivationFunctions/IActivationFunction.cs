namespace NeuralNetworks.ActivationFunctions
{
    public interface IActivationFunction
    {
        double Activate(double z);
        double ActivationDerivative(double z);
    }
}
