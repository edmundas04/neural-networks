using BenchmarkDotNet.Running;
using NeuralNetworks.ConsoleSamples.Benchmarks;
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
            Console.WriteLine("3. Digits recognition training with different settings comparison.");
            Console.WriteLine("4. XOR comparison with benchmark library");
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
                case "3":
                    new DigitsRecognitionCompareExample(1).Run();
                    break;
                case "4":
                    BenchmarkRunner.Run<LogialXORBenchmarks>();
                    break;
                default:
                    throw new ArgumentException("This menu number does not exists");
            }

            Console.Read();
        }
    }
}
