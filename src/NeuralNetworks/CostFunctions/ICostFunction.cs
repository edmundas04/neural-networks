namespace NeuralNetworks.CostFunctions
{
    public interface ICostFunction
    {
        double Cost(double output, double expectedOutput);
        double CostDerivative(double output, double expectedOutput, double activationDerivative);
    }
}
