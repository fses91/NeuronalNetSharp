using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuronalNetSharp.Import
{
    using System.IO;
    using CsvHelper;
    using MathNet.Numerics.LinearAlgebra;
    using MathNet.Numerics.LinearAlgebra.Double;

    public class MinstSmallImporter
    {
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
