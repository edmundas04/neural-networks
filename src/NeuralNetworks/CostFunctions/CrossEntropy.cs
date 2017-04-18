using System;

namespace NeuralNetworks.CostFunctions
{
    public class CrossEntropy : ICostFunction
    {
        public double Cost(double output, double expectedOutput)
        {
            return (-expectedOutput * Math.Log(output) - (1 - expectedOutput) * Math.Log(1 - output));
        }

        public double CostDerivative(double output, double expectedOutput, double activationDerivative)
        {
            return output - expectedOutput;
        }
    }
}
