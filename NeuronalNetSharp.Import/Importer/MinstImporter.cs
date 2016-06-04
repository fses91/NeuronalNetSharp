using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MathNet.Numerics.LinearAlgebra.Double;
using NeuronalNetSharp.Import.Datasets;
using NeuronalNetSharp.Import.Interfaces;

namespace NeuronalNetSharp.Import.Importer
{
    public class MinstImporter : IImportData
    {
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

                    for (var di = 0; di < numImages; di++)
                    {
                        // Initialize array.
                        
                        for (var i = 0; i < numPixels; i++)
                            pixels[i] = brData.ReadByte();

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