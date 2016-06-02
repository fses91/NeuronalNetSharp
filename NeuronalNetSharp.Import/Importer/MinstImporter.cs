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

                    // Shift position 4 bytes.
                    var test = brData.ReadInt32();

                    var numImages = 6000;
                    var rows = BitConverter.ToInt32(brData.ReadBytes(4), 0);
                    var columns = BitConverter.ToInt32(brData.ReadBytes(4).Reverse().ToArray(), 0);
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