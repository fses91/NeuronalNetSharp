namespace NeuronalNetSharp.WPF
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Windows.Media;
    using System.Windows.Media.Imaging;
    using MathNet.Numerics.LinearAlgebra;

    public static class VisualizerTmp
    {
        //public static ImageSource VisualizeLayerGrayscale(Matrix<double> matrix, int width, int height)
        //{
        //    var unit = matrix.SubMatrix(10, 1, 1, matrix.ColumnCount - 1).ToColumnWiseArray();
        //    var oldMax = unit.Max();
        //    var oldMin = unit.Min();
        //    const int newMax = 255;
        //    const int newMin = 0;

        //    for (var i = 0; i < unit.Length; i++)
        //        unit[i] = (unit[i] - oldMin)*(newMax - newMin)/(oldMax - oldMin) + newMin;

        //    var imgBytes = unit.Select(Convert.ToInt32).Select(x => (byte) x).ToArray();

        //    return CreateBitmap(imgBytes);
        //}

        public static IList<ImageSource> VisualizeLayerGrayscale(Matrix<double> layer, int width, int height)
        {
            const int newMin = 0;
            const int newMax = 255;

            var images = new List<ImageSource>();

            for (var i = 0; i < layer.RowCount; i++)
            {
                var unit = layer.SubMatrix(i, 1, 1, layer.ColumnCount - 1).ToColumnWiseArray();
                var oldMax = unit.Max();
                var oldMin = unit.Min();

                // Scale between 0 - 255.
                for (var j = 0; j < unit.Length; j++)
                    unit[j] = (unit[j] - oldMin) * (newMax - newMin) / (oldMax - oldMin) + newMin;


                var intTmp = unit.Select(Convert.ToInt32);
                var byteTmp = intTmp.Select(x => (byte) x).ToArray();
                var btmap = CreateBitmap(byteTmp, width, height);
                images.Add(btmap);
            }

            return images;
        }

        public static BitmapSource CreateBitmap(byte[] bytes, int width, int height)
        {
            return BitmapSource.Create(width, height, 96, 96, PixelFormats.Gray8, BitmapPalettes.Gray256, bytes, 28);
        }


        
    }
}