using System;

namespace NeuralNetworks.CostFunctions
{
    public class Quadratic : ICostFunction
    {
        public double Cost(double output, double expectedOutput)
        {
            return Math.Pow(output - expectedOutput, 2) / 2D;
        }

        public double CostDerivative(double output, double expectedOutput)
        {
            return output - expectedOutput;
        }
    }
}
