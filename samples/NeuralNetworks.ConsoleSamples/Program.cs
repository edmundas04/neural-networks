using NeuralNetworks.ConsoleSamples.Examples;
using System;

namespace NeuralNetworks.ConsoleSamples
{
    class Program
    {
        static void Main(string[] args)
        {
            //RunExample(new DigitsRecognitionExample());

            Run();
        }

        private static void Run()
        {
            Console.WriteLine("Menu");
            Console.WriteLine("1. Run logical XOR example.");
            Console.WriteLine("2. Run digits recognition example.");
            Console.Write("Enter example number: ");
            var exampleNumber = Console.ReadLine();

            switch (exampleNumber)
            {
                case "1":
                    RunExample(new LogicalXORExample());
                    break;
                case "2":
                    throw new NotImplementedException();
                    RunExample(new DigitsRecognitionExample());
                    break;
                default:
                    throw new ArgumentException("This menu number does not exists");
            }

            Console.Read();
        }

        private static void RunExample(IExample example)
        {
            var neuralNetwork = example.CreateNeuralNetwork();
            Console.WriteLine("Before training");
            example.DisplayEvaluation(neuralNetwork);
            example.Train(neuralNetwork);
            Console.WriteLine("After training");
            example.DisplayEvaluation(neuralNetwork);
        }
    }
}
