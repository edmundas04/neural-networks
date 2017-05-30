using System;

namespace NeuralNetworks.Extensions
{
    internal static class ArrayExtensions
    {
        internal static void Sum(this double[][] target, double[][] source)
        {
            for (int i = 0; i < source.Length; i++)
            {
                var innerArray = source[i];
                var targetArray = target[i];

                var innerArrayLegth = innerArray.Length;
                for (int j = 0; j < innerArrayLegth; j++)
                {
                    targetArray[j] += innerArray[j];
                }
            }
        }

        internal static double[][] CopyWithZeros(this double[][] source)
        {
            var length = source.Length;
            var result = new double[length][];

            for (int i = 0; i < length; i++)
            {
                var innerArrayLength = source[i].Length;
                result[i] = new double[innerArrayLength];
            }

            return result;
        }

        internal static void FillWithZeros(this double[][] arrays)
        {
            var arraysLength = arrays.Length;

            for (int i = 0; i < arraysLength; i++)
            {
                var array = arrays[i];
                var arrayLenght = array.Length;

                for (int j = 0; j < arrayLenght; j++)
                {
                    array[j] = 0;
                }
            }
        }
    }
}
