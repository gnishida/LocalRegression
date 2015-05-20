#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "MLUtils.h"
#include "LocalLinearRegression.h"
#include <fstream>

using namespace std;

int main(int argc,char *argv[]) {
	if (argc < 5) {
		cout << endl;
		cout << "Usage: " << argv[0] << " <filename of X> <filename of Y> <ratio for training data> <sigma>" << endl;
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
	ml::splitDataset(normalizedX, atof(argv[3]), trainingX, testX);
	cv::Mat_<double> trainingY, testY;
	ml::splitDataset(normalizedY, atof(argv[3]), trainingY, testY);
	
#if 1
	LocalLinearRegression llr;

	cv::Mat_<double> predY(testY.rows, testY.cols);
	for (int i = 0; i < testX.rows; ++i) {
		//cout << testX.row(i) << endl;

		cv::Mat_<double> normalized_y_hat = llr.predict(trainingX, trainingY, testX.row(i), atof(argv[4]));

		//cout << normalized_y_hat << endl;
		//cout << testY.row(i) << endl;

		normalized_y_hat.copyTo(predY.row(i));
	}
	double rmse = ml::rmse(testY, predY, true);

	ml::saveDataset("trueX.txt", testX);
	ml::saveDataset("trueY.txt", testY);
	ml::saveDataset("predY.txt", predY);

	cout << "RMSE: " << rmse << endl;
#endif

#if 0
	ofstream ofs("results.txt");
	for (double sigma = 0.1; sigma < 100; sigma += 5) {
		LocalLinearRegression llr(trainingX, trainingY, sigma);

		cv::Mat_<double> predY(testY.rows, testY.cols);
		for (int i = 0; i < testX.rows; ++i) {
			cv::Mat_<double> normalized_y_hat = llr.predict(testX.row(i));
			normalized_y_hat.copyTo(predY.row(i));
		}
		double rmse = ml::rmse(testY, predY);

		cout << sigma << "," << rmse << endl;
		ofs << sigma << "," << rmse << endl;
	}
	ofs.close();
#endif

	return 0;
}
