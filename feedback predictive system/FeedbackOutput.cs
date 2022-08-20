using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace feedback_predictive_system
{
    internal class FeedbackOutput
    {
        [ColumnName("PredictedLabel")]
        public bool IsGood { get; set; }
    }
}
