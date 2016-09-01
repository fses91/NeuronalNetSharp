namespace NeuronalNetSharp.Import
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using CsvHelper;
    using MathNet.Numerics.LinearAlgebra.Double;

    public class MinstSmallImporter
    {
        /// <summary>
        /// Imports a smaller version of the mnist dataset, which is only 20x20 pixels by picture..
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
                    var reLabel = new StreamReader(fsLabel);
                    var reData = new StreamReader(fsData);
                    var csvLabel = new CsvParser(reLabel);
                    var csvData = new CsvParser(reData);

                    const int rows = 20;
                    const int columns = 20;

                    while (true)
                    {
                        var dataRow = csvData.Read();
                        var labelRow = csvLabel.Read();

                        if (labelRow == null || dataRow == null)
                            break;


                        var dataRecord = Array.ConvertAll(dataRow, double.Parse);
                        var labelRecord = labelRow[0];

                        result.Add(new MinstDataset(DenseMatrix.OfColumnArrays(dataRecord), labelRecord, rows, columns));
                    }
                }
            }

            return result;
        }
    }
}
