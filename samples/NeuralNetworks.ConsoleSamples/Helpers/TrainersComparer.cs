using NeuralNetworks.ConsoleSamples.Extensions;
using NeuralNetworks.Tools;
using NeuralNetworks.Training;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworks.ConsoleSamples.Helpers
{
    public static class TrainersComparer
    {
        public static List<CompareResult> Compare(List<ITrainer> trainers, NeuralNetworkDto neuralNetworkDto, List<TrainingElement> trainingData, List<TrainingElement> validationData)
        {
            var result = new List<CompareResult>();

            foreach (var trainer in trainers)
            {
                var compareResult = new CompareResult();
                var neuralNetworkDtoCopy = neuralNetworkDto.Copy();
                compareResult.ElapsedTimeInMillisecond = Statistics.GetTrainingLength(trainer, neuralNetworkDtoCopy, trainingData.ToList());
                compareResult.Accuracy = Statistics.GetAccuracy(validationData, neuralNetworkDtoCopy);
                compareResult.AccuracyByMax = Statistics.GetAccuracyByMax(validationData, neuralNetworkDtoCopy);
                result.Add(compareResult);
            }

            return result;
        }
    }

    public class CompareResult
    {
        public double Accuracy { get; set; }
        public double AccuracyByMax { get; set; }
        public long ElapsedTimeInMillisecond { get; set; }
    }
}
