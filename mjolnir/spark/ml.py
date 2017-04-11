"""Backports of pyspark.ml.* from pyspark 2.x to 1.6.0"""

from pyspark.ml.feature import Bucketizer
from pyspark.ml.param.shared import Param, HasInputCol, HasOutputCol
from pyspark.ml.util import keyword_only
from pyspark.ml.wrapper import JavaEstimator


class QuantileDiscretizer(JavaEstimator, HasInputCol, HasOutputCol):
    """
    .. note:: Experimental. Backported from 2.x

    `QuantileDiscretizer` takes a column with continuous features and outputs a column with binned
    categorical features. The number of bins can be set using the :py:attr:`numBuckets` parameter.
    The bin ranges are chosen using an approximate algorithm (see the scala documentation for a
    detailed description).
    """

    @keyword_only
    def __init__(self, numBuckets=2, inputCol=None, outputCol=None, relativeError=0.001,
                 handleInvalid="error"):
        """
        __init__(self, numBuckets=2, inputCol=None, outputCol=None, relativeError=0.001, \
                 handleInvalid="error")
        """
        super(QuantileDiscretizer, self).__init__()
        self._java_obj = self._new_java_obj("org.apache.spark.ml.feature.QuantileDiscretizer",
                                            self.uid)
        self.numBuckets = Param(
            self, "numBuckets", "Maximum number of buckets (quantiles, or " +
            "categories) into which data points are grouped. Must be >= 2.")
        self._setDefault(numBuckets=2)
        kwargs = self.__init__._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, numBuckets=2, inputCol=None, outputCol=None):
        """
        setParams(self, numBuckets=2, inputCol=None, outputCol=None, relativeError=0.001, \
                  handleInvalid="error")
        Set the params for the QuantileDiscretizer
        """
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)

    def setNumBuckets(self, value):
        """
        Sets the value of :py:attr:`numBuckets`.
        """
        return self._set(numBuckets=value)

    def getNumBuckets(self):
        """
        Gets the value of numBuckets or its default value.
        """
        return self.getOrDefault(self.numBuckets)

    def _create_model(self, java_model):
        """
        Private method to convert the java_model to a Python model.
        """
        return Bucketizer(splits=list(java_model.getSplits()),
                          inputCol=self.getInputCol(),
                          outputCol=self.getOutputCol())
