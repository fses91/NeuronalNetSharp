namespace NeuronalNetSharp.Import
{
    using System.Collections.Generic;
    using System.IO;
    using MathNet.Numerics.LinearAlgebra.Double;

    public class MinstImporter
    {
        /// <summary>
        /// Imports the mnist data.
        /// </summary>
        /// <param name="dataFile">The data file.</param>
        /// <param name="labelFile">The label file.</param>
        /// <returns>The tidy dataset.</returns>
        public ICollection<IDataset> ImportData(string dataFile, string labelFile)
        {
            var result = new List<IDataset>();

            using (var fsLabel = new FileStream(labelFile, FileMode.Open))
            {
                using (var fsData = new FileStream(dataFile, FileMode.Open))
                {
                    var brData = new BinaryReader(fsData);
                    var brLabel = new BinaryReader(fsLabel);

                    brData.ReadBytes(16);
                    brLabel.ReadBytes(8);

                    const int numImages = 60000;
                    const int rows = 28;
                    const int columns = 28;
                    var numPixels = 784;
                    var pixels = new double[numPixels];

                    var newMin = -1.0;
                    var newMax = 1.0;
                    var oldMin = 0.0;
                    var oldMax = 255.0;

                    for (var di = 0; di < numImages; di++)
                    {
                        // Initialize array.
                        
                        for (var i = 0; i < numPixels; i++)
                            pixels[i] = (brData.ReadByte() - oldMin) * (newMax - newMin) / (oldMax - oldMin) + newMin;

                        var matrix = DenseMatrix.OfColumnArrays(pixels);
                        var label = brLabel.ReadByte().ToString();
                        var dataset = new MinstDataset(matrix, label, rows, columns);

                        result.Add(dataset);
                    }
                }
            }

            return result;
        }
    }
}