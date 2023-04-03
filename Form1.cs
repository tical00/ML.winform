using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Onnx;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace ML.winform
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();

            this.Load += Form1_Load;
        }

        MLContext mlContext;
        OnnxScoringEstimator onnxPredictionPipeline;
        private void Form1_Load(object sender, EventArgs e)
        {
            mlContext = new MLContext();

            string onnxPath = Path.Combine(Application.StartupPath, @"Assets\Network.onnx");
            onnxPredictionPipeline = mlContext.Transforms.ApplyOnnxModel(onnxPath);
        }

        public class OnnxInput
        {
            private float _sepal_length = 1.0f;
            [ColumnName("Sepal_Length")]
            public float Sepal_Length
            {
                get
                {
                    return _sepal_length;
                }
                set
                {
                    // validate range [1,10]
                    if (value >= 1 && value <= 10)
                    {
                        _sepal_length = value;
                    }
                }
            }

            private float _sepal_width = 1.0f;
            [ColumnName("Sepal_Width")]
            public float Sepal_Width
            {
                get
                {
                    return _sepal_width;
                }
                set
                {
                    // validate range [1, 8]
                    if (value >= 1 && value <= 8)
                    {
                        _sepal_width = value;
                    }
                }
            }

            private float _petal_length = 0.5f;
            [ColumnName("Petal_Length")]
            public float Petal_Length
            {
                get
                {
                    return _petal_length;
                }
                set
                {
                    // validate range [0.5, 10]
                    if (value >= 0.5 && value <= 10)
                    {
                        _petal_length = value;
                    }
                }
            }

            private float _petal_width = 0.1f;
            [ColumnName("Petal_Width")]
            public float Petal_Width
            {
                get
                {
                    return _petal_width;
                }
                set
                {
                    // validate range [0.1, 5]
                    if (value >= 0.1 && value <= 5)
                    {
                        _petal_width = value;
                    }
                }
            }
        }

        public class OnnxOutput
        {
            [ColumnName("output")]
            public string PredictedIris { get; set; }
        }
        private string[] _labels = { "Iris-setosa", "Iris-versiocolor", "Iris-verginica" };

        private void input_ValueChanged(object sender, EventArgs e)
        {
            
            if (mlContext != null) 
            {
                var input = new OnnxInput
                {
                    Sepal_Length = (float)sepal_length_input.Value / 10.0f,
                    Sepal_Width = (float)sepal_width_input.Value / 10.0f,
                    Petal_Length = (float)petal_length_input.Value / 10.0f,
                    Petal_Width = (float)petal_length_input.Value / 10.0f
                };

                var data = GetInputData(input, mlContext);
                var transformedValues = onnxPredictionPipeline.Fit(data).Transform(data);
                var predictions = mlContext.Data.CreateEnumerable<Prediction>(
                                transformedValues, reuseRowObject: false);

                string strResult = "";
                
                int max_index = 0;
                float max_value = 0;
     
                foreach (var item in predictions)
                {
                    int num = 0;
                    foreach (var score in item.output.Take(outputSize))
                    {
                        if (score > max_value)
                        {
                            max_value = score;
                            max_index = num;
                        }
                        strResult += $"Case #{num+1} {_labels.ElementAt(num)} Score: {score}\n";
                        num++;
                    }
                }

                result.Text = strResult;
                model_output.Text = _labels.ElementAt(max_index);
            }
        }

        private const int inputSize = 1 * 4;
        public class TensorData
        {
            [VectorType(inputSize)]
            public float[] input { get; set; }
        }
        public static IDataView GetInputData(OnnxInput _input, MLContext context)
        {
            float[] data = new float[4];
            data[0] = _input.Sepal_Length;
            data[1] = _input.Sepal_Width;
            data[2] = _input.Petal_Length;
            data[3] = _input.Petal_Width;
            return context.Data.LoadFromEnumerable(new TensorData[] {
                new TensorData { input = data }
            });
        }

        private const int outputSize = 1 * 3;
        class Prediction
        {
            [VectorType(outputSize)]
            public float[] output { get; set;}
        }

    }
}
