using NeuralNetworks.ConsoleSamples.Examples;
using System;

namespace NeuralNetworks.ConsoleSamples
{
    class Program
    {
        static void Main(string[] args)
        {
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
                    new LogicalXORExample().Run();
                    break;
                case "2":
                    new DigitsRecognitionExample(30).Run();
                    break;
                default:
                    throw new ArgumentException("This menu number does not exists");
            }

            Console.Read();
        }
    }
}
