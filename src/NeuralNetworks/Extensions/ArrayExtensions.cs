namespace NeuralNetworks.Extensions
{
    internal static class ArrayExtensions
    {
        internal static void Sum(this double[][] target, double[][] source)
        {
            for (int i = 0; i < source.Length; i++)
            {
                for (int j = 0; j < source[i].Length; j++)
                {
                    target[i][j] += source[i][j];
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
    }
}
