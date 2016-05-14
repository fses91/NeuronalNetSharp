using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NeuronalNetSharp.Import.Datasets;
using NeuronalNetSharp.Import.Interfaces;

namespace NeuronalNetSharp.Import.Importer
{
    public class MinstImporter : IImportData<ICollection<IDataset<int[][]>>>
    {
        public ICollection<IDataset<int[][]>> ImportData(string dataFile, string labelFile, int amountData)
        {
            var result = new List<IDataset<int[][]>>();

            using (var fsLabel = new FileStream(labelFile, FileMode.Open))
            {
                using (var fsData = new FileStream(dataFile, FileMode.Open))
                {
                    var brData = new BinaryReader(fsData);
                    var brLabel = new BinaryReader(fsLabel);

                    // Shift position 4 bytes.
                    brData.ReadBytes(4);
                    var numImages = BitConverter.ToInt32(brData.ReadBytes(4).Reverse().ToArray(), 0);
                    var rows = BitConverter.ToInt32(brData.ReadBytes(4).Reverse().ToArray(), 0);
                    var columns = BitConverter.ToInt32(brData.ReadBytes(4).Reverse().ToArray(), 0);

                    for (var di = 0; di < numImages; di++)
                    {
                        // Initialize array.
                        var pixels = InitializeArray(rows, columns);

                        for (var i = 0; i < rows; i++)
                        {
                            for (var j = 0; j < columns; j++)
                            {
                                var b = (int)brData.ReadByte();
                                pixels[i][j] = b;
                            }
                        }

                        var lbl = brLabel.ReadByte().ToString();
                        var dataset = new MinstDataset { Data = pixels, Label = lbl };
                        result.Add(dataset);
                    }
                }
                
            }

            return result;
        }

        private static int[][] InitializeArray(int rows, int columns)
        {
            var array = new int[rows][];
            for (var i = 0; i < array.Length; ++i)
                array[i] = new int[columns];

            return array;
        }
    }
}