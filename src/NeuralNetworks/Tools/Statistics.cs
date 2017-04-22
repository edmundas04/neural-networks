using NeuralNetworks.Training;
using System.Collections.Generic;
using System.Diagnostics;

namespace NeuralNetworks.Tools
{
    public static class Statistics
    {
        public static long GetTrainingLength(ITrainer trainer, NeuralNetworkDto neuralNetworkDto, List<TrainingElement> trainingData)
        {
            var stopWatch = new Stopwatch();
            stopWatch.Start();
            trainer.Train(neuralNetworkDto, trainingData);
            stopWatch.Stop();
            return stopWatch.ElapsedMilliseconds;
        }

        public static double GetAccuracyByMax(List<TrainingElement> validationData, NeuralNetworkDto neuralNetworkDto)
        {
            var neuralNetwork = new NeuralNetwork(neuralNetworkDto);

            var correctCount = 0;

            foreach (var validationItem in validationData)
            {
                var output = neuralNetwork.Run(validationItem.Inputs);
                if (CheckOutput(output, validationItem.ExpectedOutputs))
                {
                    correctCount++;
                }
            }

            return (((double)correctCount) / ((double)validationData.Count)) * 100D;
        }

        private static bool CheckOutput(double[] output, double[] expectedOutput)
        {
            var maxIndex = 0;
            var maxValue = output[0];

            for (int i = 1; i < output.Length; i++)
            {
                if (output[i] > maxValue)
                {
                    maxValue = output[i];
                    maxIndex = i;
                }
            }

            return expectedOutput[maxIndex] == 1D;
        }

        public static double GetAccuracy(List<TrainingElement> validationData, NeuralNetworkDto neuralNetworkDto)
        {
            var neuralNetwork = new NeuralNetwork(neuralNetworkDto);
            var correctCount = 0;

            for (int i = 0; i < validationData.Count; i++)
            {
                var validationElement = validationData[i];
                var outputs = neuralNetwork.Run(validationElement.Inputs);
                var isCorrect = true;

                for (int j = 0; j < outputs.Length; j++)
                {
                    var outputValue = outputs[j] < 0.5D ? 0 : 1;
                    if (outputValue != validationElement.ExpectedOutputs[j])
                    {
                        isCorrect = false;
                        break;
                    }
                }

                if (isCorrect)
                {
                    correctCount++;
                }

            }

            return (((double)correctCount) / ((double)validationData.Count)) * 100D;
        }
    }
}
