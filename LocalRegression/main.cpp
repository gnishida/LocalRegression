#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "MLUtils.h"
#include "LocalLinearRegression.h"
#include <fstream>

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

	ml::addBias(normalizedX);

	cv::Mat_<double> trainingX, testX;
	ml::splitDataset(normalizedX, 0.9, trainingX, testX);
	cv::Mat_<double> trainingY, testY;
	ml::splitDataset(normalizedY, 0.9, trainingY, testY);
	
#if 0
	LocalLinearRegression llr(trainingX, trainingY, atof(argv[3]));

	double rmse = 0.0;
	for (int i = 0; i < testX.rows; i += 10) {
		cv::Mat_<double> normalized_y_hat = llr.predict(testX.row(i));
		rmse += sqrt(ml::mat_sum(ml::mat_square(testY.row(i) - normalized_y_hat)));
	}
	rmse /= testX.rows / 10;

	cout << rmse << endl;
#endif

#if 1
	ofstream ofs("results2.txt");
	for (double sigma = 0.01; sigma < 0.1; sigma += 0.01) {
		LocalLinearRegression llr(trainingX, trainingY, sigma);

		double rmse = 0.0;
		for (int i = 0; i < testX.rows; ++i) {
			cv::Mat_<double> normalized_y_hat = llr.predict(testX.row(i));
			rmse += sqrt(ml::mat_sum(ml::mat_square(testY.row(i) - normalized_y_hat)));
		}
		rmse /= testX.rows;

		cout << sigma << "," << rmse << endl;
		ofs << sigma << "," << rmse << endl;
	}
	ofs.close();
#endif

	return 0;
}
