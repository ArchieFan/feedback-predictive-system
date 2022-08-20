using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace feedback_predictive_system
{
    internal class FeedbackTrainingData
    {
        [LoadColumn(0), ColumnName("Label")]
        public bool IsGood { get; set; }

        [LoadColumn(1)]
        public string? FeedBackText { get; set; }




    }
}
