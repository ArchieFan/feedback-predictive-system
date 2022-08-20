using System;
using System.Net.Http.Headers;
using Microsoft.ML;


namespace feedback_predictive_system
{
    static class Program
    {
        static List<FeedbackTrainingData> feedbackTrainingDatas = 
            new List<FeedbackTrainingData>();

        static List<FeedbackTrainingData> testingDatas =
            new List<FeedbackTrainingData>();

        static void LoadTrainingData()
        {
            feedbackTrainingDatas.Add(new FeedbackTrainingData()
            {
                FeedBackText = "this is good", 
                IsGood = true
            });
            feedbackTrainingDatas.Add(new FeedbackTrainingData()
            {
                FeedBackText = "this is horrible",
                IsGood = false
            });
            feedbackTrainingDatas.Add(new FeedbackTrainingData()
            {
                FeedBackText = "good",
                IsGood = true
            });
            feedbackTrainingDatas.Add(new FeedbackTrainingData()
            {
                FeedBackText = "nice",
                IsGood = true
            });
            feedbackTrainingDatas.Add(new FeedbackTrainingData()
            {
                FeedBackText = "bad",
                IsGood = false
            });
        }

        static void LoadTestData()
        {
            testingDatas.Add(new FeedbackTrainingData()
            {
                FeedBackText = "good",
                IsGood = true
            });
            testingDatas.Add(new FeedbackTrainingData()
            {
                FeedBackText = "bad",
                IsGood = false
            });
            testingDatas.Add(new FeedbackTrainingData()
            {
                FeedBackText = "bad",
                IsGood = false
            });
            testingDatas.Add(new FeedbackTrainingData()
            {
                FeedBackText = "bad",
                IsGood = false
            });
            testingDatas.Add(new FeedbackTrainingData()
            {
                FeedBackText = "bad",
                IsGood = false
            });
            testingDatas.Add(new FeedbackTrainingData()
            {
                FeedBackText = "bad",
                IsGood = false
            });
            testingDatas.Add(new FeedbackTrainingData()
            {
                FeedBackText = "bad",
                IsGood = false
            });
            testingDatas.Add(new FeedbackTrainingData()
            {
                FeedBackText = "bad",
                IsGood = false
            });
            testingDatas.Add(new FeedbackTrainingData()
            {
                FeedBackText = "bad",
                IsGood = false
            });
            testingDatas.Add(new FeedbackTrainingData()
            {
                FeedBackText = "bad",
                IsGood = false
            });
            testingDatas.Add(new FeedbackTrainingData()
            {
                FeedBackText = "bad",
                IsGood = false
            });
            testingDatas.Add(new FeedbackTrainingData()
            {
                FeedBackText = "bad",
                IsGood = false
            });
            testingDatas.Add(new FeedbackTrainingData()
            {
                FeedBackText = "bad",
                IsGood = false
            });
        }
        static void Main()
        {
            //Step 1 :- we need to load the training data
            LoadTrainingData();
            // Step 2 :- create object pf MLContext
            var mlContext = new MLContext();
            // Step 3 :- Convert your data into IDataView
            IDataView dataView =
                mlContext.Data.LoadFromEnumerable<FeedbackTrainingData>(feedbackTrainingDatas);
            // Step 4 :- we need to create a pipeline and define the workflow in it
            var _model = mlContext.BinaryClassification.Trainers.FastTree( numberOfLeaves: 50,
                                      numberOfTrees: 50,
                                      learningRate: 1, minimumExampleCountPerLeaf: 1);
            var pipeline =
                mlContext.Transforms.Text.FeaturizeText("Features","FeedBackText")
                .Append(_model);
            // Step 5 :- Training the algorithm and we want the model out
            var model = pipeline.Fit(dataView);
            // Step 6 :- load the test data and run the test data to check our models accuracy
            LoadTestData();
            IDataView testdataView =
                mlContext.Data.LoadFromEnumerable<FeedbackTrainingData>(testingDatas);
            var predictions = model.Transform(testdataView);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions,"Label");
            Console.WriteLine($"Accuracy : {metrics.Accuracy}");

            // Step 7 :- use the model
            var feedbackInput = new FeedbackTrainingData();
            var predictionFunction = mlContext.Model.CreatePredictionEngine<FeedbackTrainingData, FeedbackOutput>(model);
            while (feedbackInput.FeedBackText != "exit")
            {
                Console.WriteLine("Enter a feedback string");
                string feedbackstring = Console.ReadLine();
                feedbackInput.FeedBackText = feedbackstring;
                var prediction = predictionFunction.Predict(feedbackInput);
                Console.WriteLine($"Predicted :- {prediction.IsGood} ");
            }


            Console.ReadLine();
        }
    }
}



