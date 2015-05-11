#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "MLUtils.h"
#include "LocalLinearRegression.h"

using namespace std;

int main(int argc,char *argv[]) {
	if (argc < 3) {
		cout << endl;
		cout << "Usage: " << argv[0] << " <filename of X> <filename of Y>" << endl;
		cout << endl;

		return -1;
	}

	cv::Mat_<double> X, Y;
	ml::loadDataset(argv[1], X);
	ml::loadDataset(argv[2], Y);

	cv::Mat_<double> normalizedX, meanX, stddevX;
	ml::normalizeDataset(X, normalizedX, meanX, stddevX);
	cv::Mat_<double> normalizedY, meanY, stddevY;
	ml::normalizeDataset(Y, normalizedY, meanY, stddevY);

	cv::Mat_<double> trainingX, testX;
	ml::splitDataset(normalizedX, 0.9, trainingX, testX);
	cv::Mat_<double> trainingY, testY;
	ml::splitDataset(normalizedY, 0.9, trainingY, testY);


	LocalLinearRegression llr(trainingX, trainingY, 0.2);

	double rmse = 0.0;
	for (int i = 0; i < testX.rows; ++i) {
		cv::Mat_<double> normalized_y_hat = llr.predict(testX.row(i));
		rmse += sqrt(ml::mat_sum(ml::mat_square(testY.row(i) - normalized_y_hat)));
	}
	rmse /= testX.rows;

	cout << rmse << endl;

	return 0;
}
